"""Conservative unique-token receive bound; no observed-route percentile shortcut."""

def validate_bound(capacity,ep_size,send_rows):
    if capacity not in (8,16) or ep_size!=4 or len(send_rows)!=ep_size:raise ValueError('Unsupported capacity/topology')
    if any(not isinstance(x,int) or x<0 or x>capacity//ep_size for x in send_rows):raise ValueError('Send row bound violated')
    worst_receive=sum(send_rows)
    if worst_receive>capacity:raise ValueError('Receive bound exceeded')
    return worst_receive
