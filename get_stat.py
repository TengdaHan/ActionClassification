from backbone.select_backbone import select_backbone

def show_num_params(net_name):
    model, _ = select_backbone(net_name)
    num_param = sum(p.numel() for p in model.parameters())
    return '%s\t%d' % (net_name, num_param)

net_list = ['r18', 'r34', 'r50',
            'r3d18', 'r3d34', 'r3d50',
            'i3d', 's3d', 's3dg']

print('=== number of parameters ===')
for net in net_list:
    print(show_num_params(net))
print('============================')