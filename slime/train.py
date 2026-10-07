import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking
from slime.utils.misc import should_run_periodic_action
from slime.utils.rl_iteration_timer import RLIterationTimer
from slime.utils.p1_profile import start_rollout_profile


def train(args):
    configure_logger()
    release_train = args.release_train
    iteration_timer = RLIterationTimer(args)

    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout and not release_train:
        ray.get(rollout_manager.onload_weights.remote())

    # Always push actor weights to rollout once weights are loaded.
    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    def offload_train(actor_trains_this_step):
        # Each model auto-offloads after train() when offload_train is set,
        # so we only need clear_memory for the non-offload case.
        if not args.offload_train:
            if not args.use_critic or actor_trains_this_step:
                actor_model.clear_memory()
            else:
                critic_model.clear_memory()

    # train loop.
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))

        start_rollout_profile(rollout_manager, rollout_id)
        iteration_timer.start(rollout_id)
        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
        iteration_timer.mark("rollout_reward_data")

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())
        iteration_timer.mark("rollout_offload")

        if release_train:
            actor_model.create()

        actor_trains = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
        if args.use_critic:
            value_refs = critic_model.async_train(rollout_id, rollout_data_ref)
            if actor_trains:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref, external_data=value_refs))
            else:
                ray.get(value_refs)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        iteration_timer.mark("training_rpc_including_internal_transfers")

        if release_train or should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        ):
            force_sync = release_train or rollout_id == args.num_rollout - 1
            if actor_trains:
                actor_model.save_model(rollout_id, force_sync=force_sync)
            if args.use_critic:
                critic_model.save_model(rollout_id, force_sync=force_sync)
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))

        offload_train(actor_trains)
        iteration_timer.mark("post_train_clear")
        if args.offload_rollout and not release_train:
            ray.get(rollout_manager.onload_weights.remote())
        iteration_timer.mark("rollout_weights_onload")
        actor_model.update_weights()
        iteration_timer.mark("weight_sync")

        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())
        iteration_timer.mark("rollout_kv_onload")
        iteration_timer.finish()

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
