from darknet_meta import Darknet
import dataset
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
from utils import *
from cfg import cfg
from cfg import parse_cfg
import os
import pdb


def valid(datacfg, darknetcfg, learnetcfg, weightfile, outfile, use_baserw=False):
    options = read_data_cfg(datacfg)
    valid_images = options['valid']
    metadict = options['meta']
    # name_list = options['names']
    # backup = cfg.backup
    ckpt = weightfile.split('/')[-1].split('.')[0]
    backup = weightfile.split('/')[-2]
    ckpt_pre = '/ene_' if use_baserw else '/ene' # 是否使用base reweighting weights
    prefix = 'results/' + backup.split('/')[-1] + ckpt_pre + ckpt # 文件保存路径
    print('saving to: ' + prefix)
    # prefix = 'results/' + weightfile.split('/')[1]
    # names = load_class_names(name_list)

    with open(valid_images) as fp:
        tmp_files = fp.readlines()
        valid_files = [item.rstrip() for item in tmp_files]
    
    # 构建网络
    m = Darknet(darknetcfg, learnetcfg) # model
    m.print_network()
    # m.load_weights(weightfile) # 因为会报错（原因未知），导致无法调试，所以暂时注释掉
    m.cuda()
    m.eval()

    # 读取query set
    valid_dataset = dataset.listDataset(valid_images, shape=(m.width, m.height),
                       shuffle=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ]))
    valid_batchsize = 2
    assert(valid_batchsize > 1)

    kwargs = {'num_workers': 4, 'pin_memory': True}
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=valid_batchsize, shuffle=False, **kwargs) 


    if False:
        metaset = dataset.MetaDataset(metafiles=metadict, train=False, ensemble=True)
        metaloader = torch.utils.data.DataLoader(
            metaset,
            batch_size=len(metaset),
            shuffle=False,
            **kwargs
        )
        metaloader = iter(metaloader)
        n_cls = len(metaset.classes)

        print('===> Generating dynamic weights...')
        metax, mask = metaloader.next()
        metax, mask = metax.cuda(), mask.cuda()
        metax, mask = Variable(metax, volatile=True), Variable(mask, volatile=True)
        dynamic_weights = m.meta_forward(metax, mask)

        for i in range(len(dynamic_weights)):
            assert dynamic_weights[i].size(0) == sum(metaset.meta_cnts)
            inds = np.cumsum([0] + metaset.meta_cnts)
            new_weight = []
            for j in range(len(metaset.meta_cnts)):
                new_weight.append(torch.mean(dynamic_weights[i][inds[j]:inds[j+1]], dim=0))
            dynamic_weights[i] = torch.stack(new_weight)
            print(dynamic_weights[i].shape)
    else:
        ### IMPORTANT：获取reweighting weights
        # 读取support set
        metaset = dataset.MetaDataset(metafiles=metadict, train=False, ensemble=True, with_ids=True)
        metaloader = torch.utils.data.DataLoader(
            metaset,
            batch_size=64,
            shuffle=False,
            **kwargs
        )
        # metaloader = iter(metaloader)
        n_cls = len(metaset.classes) # 类别数量

        enews = [0.0] * n_cls
        cnt = [0.0] * n_cls # 整个support set中各个class有几张图片，用来计算各个class对应的reweighting weights的均值
        print('===> Generating dynamic weights...')
        kkk = 0 # 第几个batch
        for metax, mask, clsids in metaloader: # 遍历每个batch
            # metax：形状为[64, 3, 416, 416] (batch_size, C, H, W)
            # mask：形状为[64, 1, 416, 416] (batch_size, C, H, W)
            # clsids：形状为[64]  (batch_size,) 表示64张图片的class
            print('===> {}/{}'.format(kkk, len(metaset) // 64)) # 第几个batch/共有几个batch
            kkk += 1
            metax, mask = metax.cuda(), mask.cuda()
            metax, mask = Variable(metax, volatile=True), Variable(mask, volatile=True)
            dws = m.meta_forward(metax, mask)
            dw = dws[0] # 估计形状为[64, 1024]  (batch_size, 1024)
            for ci, c in enumerate(clsids):
                # ci：该图片在该batch中的index
                # c：该图片的class
                enews[c] = enews[c] * cnt[c] / (cnt[c] + 1) + dw[ci] / (cnt[c] + 1) # 更新该class的reweighting weights
                cnt[c] += 1
        dynamic_weights = [torch.stack(enews)]

        # import pickle
        # with open('data/rws/voc_novel2_.pkl', 'wb') as f:
        #     tmp = [x.data.cpu().numpy() for x in dynamic_weights]
        #     pickle.dump(tmp, f)
        # import pdb; pdb.set_trace()

        if use_baserw:
            import pickle
            # f = 'data/rws/voc_novel{}_.pkl'.format(cfg.novelid)
            f = 'data/rws/voc_novel{}_.pkl'.format(0)
            print('===> Loading from {}...'.format(f))
            with open(f, 'rb') as f:
            # with open('data/rws/voc_novel0_.pkl', 'rb') as f:
                rws = pickle.load(f)
                rws = [Variable(torch.from_numpy(rw)).cuda() for rw in rws]
                tki = cfg._real_base_ids
                for i in range(len(rws)):
                    dynamic_weights[i][tki] = rws[i][tki]
                    # dynamic_weights[i] = rws[i]
            # pdb.set_trace()


    if not os.path.exists(prefix):
        # os.mkdir(prefix)
        os.makedirs(prefix)

    fps = [0]*n_cls
    for i, cls_name in enumerate(metaset.classes):
        buf = '%s/%s%s.txt' % (prefix, outfile, cls_name)
        fps[i] = open(buf, 'w')
   
    lineId = -1
    
    conf_thresh = 0.005
    nms_thresh = 0.45
    for batch_idx, (data, target) in enumerate(valid_loader):
        data = data.cuda()
        data = Variable(data, volatile = True)
        output = m.detect_forward(data, dynamic_weights)

        if isinstance(output, tuple):
            output = (output[0].data, output[1].data)
        else:
            output = output.data

        # import pdb; pdb.set_trace()
        batch_boxes = get_region_boxes_v2(output, n_cls, conf_thresh, m.num_classes, m.anchors, m.num_anchors, 0, 1)

        if isinstance(output, tuple):
            bs = output[0].size(0)
        else:
            assert output.size(0) % n_cls == 0
            bs = output.size(0) // n_cls

        for b in range(bs):
            lineId = lineId + 1
            imgpath = valid_dataset.lines[lineId].rstrip()
            print(imgpath)
            imgid = os.path.basename(imgpath).split('.')[0]
            width, height = get_image_size(imgpath)
            for i in range(n_cls):
                # oi = i * bs + b
                oi = b * n_cls + i
                boxes = batch_boxes[oi]
                boxes = nms(boxes, nms_thresh)
                for box in boxes:
                    x1 = (box[0] - box[2]/2.0) * width
                    y1 = (box[1] - box[3]/2.0) * height
                    x2 = (box[0] + box[2]/2.0) * width
                    y2 = (box[1] + box[3]/2.0) * height

                    det_conf = box[4]
                    for j in range((len(box)-5)/2):
                        cls_conf = box[5+2*j]
                        cls_id = box[6+2*j]
                        prob =det_conf * cls_conf
                        fps[i].write('%s %f %f %f %f %f\n' % (imgid, prob, x1, y1, x2, y2))

    for i in range(n_cls):
        fps[i].close()

    # import pdb; pdb.set_trace()

if __name__ == '__main__':
    import sys
    """python valid_ensemble.py cfg/metatune.data cfg/darknet_dynamic.cfg cfg/reweighting_net.cfg path/to/tuned/weightfile"""
    if len(sys.argv) in [5,6,7]:
        datacfg = sys.argv[1]
        darknet = parse_cfg(sys.argv[2])
        learnet = parse_cfg(sys.argv[3])
        weightfile = sys.argv[4]
        if len(sys.argv) >= 6: # 设置GPU
            gpu = sys.argv[5]
        else:
            gpu = '0'
        if len(sys.argv) == 7: # 设置use_baserw
            use_baserw = True
        else:
            use_baserw = False

        data_options  = read_data_cfg(datacfg)
        net_options   = darknet[0]
        meta_options  = learnet[0]
        data_options['gpus'] = gpu
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu

        # Configure options
        cfg.config_data(data_options)
        cfg.config_meta(meta_options)
        cfg.config_net(net_options)

        outfile = 'comp4_det_test_'
        valid(datacfg, darknet, learnet, weightfile, outfile, use_baserw)
    else:
        print('Usage:')
        print(' python valid.py datacfg cfgfile weightfile')
