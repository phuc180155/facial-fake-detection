import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
import torch
import torch.nn as nn
import argparse
import json
from torchsummary import summary

def parse_args():
    parser = argparse.ArgumentParser(description="Deepfake detection")
    parser.add_argument('--train_dir', type=str, default="", help="path to train data")
    parser.add_argument('--val_dir', type=str, default="", help="path to validation data")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size")
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--n_epochs', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--image_size', type=int, default=128, help='the height / width of the input image to network')
    parser.add_argument('--workers', type=int, default=0, help='number wokers for dataloader ')
    parser.add_argument('--checkpoint',default = None,required=True, help='path to checkpoint ')
    parser.add_argument('--gpu_id',type=int, default = 0, help='GPU id ')
    parser.add_argument('--resume',type=str, default = '', help='Resume from checkpoint ')
    parser.add_argument('--loss',type=str, default = "bce", help='Loss function use')
    parser.add_argument('--gamma',type=float, default=0.0, help="gamma hyperparameter for focal loss")
    parser.add_argument('--eval_per_iters',type=int, default=-1, help='Evaluate per some iterations')
    parser.add_argument('--es_metric',type=str, default='val_loss', help='Criterion for early stopping')
    parser.add_argument('--es_patience',type=int, default=5, help='Early stopping epoch')
    
    sub_parser = parser.add_subparsers(dest="model", help="Choose model of the available ones:  xception, efficient_dual, ViT, CrossViT, efficient ViT, dual efficient vit...")
    
    ######################## CNN architecture:
    parser_xception = sub_parser.add_parser('xception', help='XceptionNet')
    parser_dual_eff = sub_parser.add_parser('dual_efficient', help="Efficient-Frequency Net")
    # Ablation study
    parser_dual_attn_eff = sub_parser.add_parser('dual_attn_efficient', help="Ablation Study")
    parser_dual_attn_eff.add_argument("--patch_size",type=int,default=7,help="patch_size")
    parser_dual_attn_eff.add_argument("--version",type=str, default="cross_attention-freq-add", required=True, help="Some changes in model")
    parser_dual_attn_eff.add_argument("--weight", type=float, default=1, help="Weight for frequency vectors")
    parser_dual_attn_eff.add_argument("--freeze", type=int, default=0, help="Freeze backbone")
    
    ######################## Vision transformer architecture:
    parser.add_argument('--dim',type=int, default = 1024, help='dim of embeding')
    parser.add_argument('--depth',type=int, default = 6, help='Number of attention layer in transformer module')
    parser.add_argument('--heads',type=int, default = 8, help='number of head in attention layer')
    parser.add_argument('--mlp_dim',type=int, default = 2048, help='dim of hidden layer in transformer layer')
    parser.add_argument('--dim_head',type=int, default = 64, help='in transformer layer ')
    
    # ViT
    parser_vit = sub_parser.add_parser('vit', help='ViT transformer Net')
    # Efficient ViT (CViT)
    parser_efficientvit = sub_parser.add_parser('efficient_vit', help='CrossViT transformer Net')
    parser_efficientvit.add_argument("--patch_size",type=int,default=7,help="patch_size in vit")
    # SwinViT
    parser_swim_vit = sub_parser.add_parser('swin_vit', help='Swim transformer')
    
    # My refined model:
    parser_dual_eff_vit = sub_parser.add_parser('dual_efficient_vit', help='My model')
    parser_dual_eff_vit.add_argument("--patch_size",type=int,default=7,help="patch_size in vit")
    parser_dual_eff_vit.add_argument("--version",type=str, default="cross_attention-freq-add", required=True, help="Some changes in model")
    parser_dual_eff_vit.add_argument("--weight", type=float, default=1, help="Weight for frequency vectors")
    parser_dual_eff_vit.add_argument("--freeze", type=int, default=0, help="Freeze backbone")
    
    parser_dual_eff_vit_v2 = sub_parser.add_parser('dual_efficient_vit_v2', help='My model')
    parser_dual_eff_vit_v2.add_argument("--patch_size",type=int,default=7,help="patch_size in vit")
    parser_dual_eff_vit_v2.add_argument("--version",type=str, default="cross_attention-freq-add", required=True, help="Some changes in model")
    parser_dual_eff_vit_v2.add_argument("--weight", type=float, default=1, help="Weight for frequency vectors")
    parser_dual_eff_vit_v2.add_argument("--freeze", type=int, default=0, help="Weight for frequency vectors")
    
    ############# adjust image
    parser.add_argument('--adj_brightness',type=float, default = 1, help='adj_brightness')
    parser.add_argument('--adj_contrast',type=float, default = 1, help='adj_contrast')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args)
    
    model = args.model
    # Config device
    gpu_id = 0 if int(args.gpu_id) >=0 else -1
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    print(gpu_id)
    
    # Adjustness:
    adj_brightness = float(args.adj_brightness)
    adj_contrast = float(args.adj_contrast)
    
    # Save args to text:
    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)
    # with open(os.path.join(args.checkpoint, 'args.txt'), 'w') as f:
    #     json.dump(args.__dict__, f, indent=2)
        
    ################# TRAIN #######################
    if model == "xception":
        from module.train_torch import train_image_stream
        from model.cnn.xception import xception
        model = xception(pretrained=True)
        args_txt = "imgsize_{}_lr_{}_batchsize_{}_es_{}_loss_{}".format(args.image_size, args.lr, args.batch_size, args.es_metric, args.loss)
        criterion = [args.loss]
        if args.gamma:
            args_txt += "gamma_{}".format(args.gamma)
            criterion.append(args.gamma)
            
        train_image_stream(model, criterion_name=criterion, train_dir=args.train_dir, val_dir=args.val_dir, image_size=args.image_size, lr=args.lr,\
                           batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, resume=args.resume, epochs=args.n_epochs, eval_per_iters=args.eval_per_iters,\
                           adj_brightness=adj_brightness, adj_contrast=adj_contrast, es_metric=args.es_metric, es_patience=args.es_patience, model_name="exception", args_txt=args_txt)
    
    elif model == "dual_efficiet":
        from module.train_torch import train_dual_stream
        from model.cnn.dual_efficient import DualEfficient
        
        model = DualEfficient()
        args_txt = "imgsize_{}_lr_{}_batchsize_{}_es_{}_loss_{}".format(args.image_size, args.lr, args.batch_size, args.es_metric,args.loss)
        criterion = [args.loss]
        if args.gamma:
            args_txt += "gamma_{}".format(args.gamma)
            criterion.append(args.gamma)
            
        train_dual_stream(model, criterion_name=criterion, train_dir=args.train_dir, val_dir=args.val_dir, image_size=args.image_size, lr=args.lr,\
                           batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, resume=args.resume, epochs=args.n_epochs, eval_per_iters=args.eval_per_iters,\
                           adj_brightness=adj_brightness, adj_contrast=adj_contrast, es_metric=args.es_metric, es_patience=args.es_patience, model_name="dual-efficient", args_txt=args_txt)

    elif model == "dual_attn_efficient":
        from module.train_torch import train_dual_stream
        from model.cnn.dual_crossattn_efficient import DualCrossAttnEfficient
        
        dropout = 0.15
        model = DualCrossAttnEfficient(
            image_size=args.image_size,
            patch_size=args.patch_size,
            num_classes=1,
            dim=args.dim,
            mlp_dim=args.mlp_dim,
            dropout=dropout,
            version=args.version,
            weight=args.weight,
            freeze=args.freeze
        )
        
        args_txt = "ver_{}_weight_{}_imgsize_{}_lr_{}_patchsize_{}_es_{}_loss_{}_freeze_{}".format(args.version, args.weight, args.image_size, args.lr, args.patch_size, args.es_metric, args.loss, args.freeze)
        criterion = [args.loss]
        if args.gamma:
            args_txt += "gamma_{}".format(args.gamma)
            criterion.append(args.gamma)
        
        train_dual_stream(model, criterion_name=criterion, train_dir=args.train_dir, val_dir=args.val_dir, image_size=args.image_size, lr=args.lr,\
                           batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, resume=args.resume, epochs=args.n_epochs, eval_per_iters=args.eval_per_iters,\
                           adj_brightness=adj_brightness, adj_contrast=adj_contrast, es_metric=args.es_metric, es_patience=args.es_patience, model_name="dual-attn-efficient", args_txt=args_txt)
        
    elif model == "efficient_vit":
        from module.train_torch import train_image_stream
        from model.vision_transformer.efficient_vit import EfficientViT

        dropout = 0.15
        emb_dropout = 0.15
        model = EfficientViT(
            selected_efficient_net=0,
            image_size=args.image_size,
            patch_size=args.patch_size,
            num_classes=1,
            dim=args.dim,
            depth=args.depth,
            heads=args.heads,
            mlp_dim=args.mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout,
        )
        
        args_txt = "imgsize_{}_lr_{}_patchsize_{}_es_{}_loss_{}".format(args.image_size, args.lr, args.patch_size, args.es_metric, args.loss)
        criterion = [args.loss]
        if args.gamma:
            args_txt += "gamma_{}".format(args.gamma)
            criterion.append(args.gamma)
        
        train_image_stream(model, criterion_name=criterion, train_dir=args.train_dir, val_dir=args.val_dir, image_size=args.image_size, lr=args.lr,\
                           batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, resume=args.resume, epochs=args.n_epochs, eval_per_iters=args.eval_per_iters,\
                           adj_brightness=adj_brightness, adj_contrast=adj_contrast, es_metric=args.es_metric, es_patience=args.es_patience, model_name="exception", args_txt=args_txt)
        
    elif model == "dual_efficient_vit":
        from module.train_torch import train_dual_stream
        from model.vision_transformer.dual_efficient_vit import DualEfficientViT
        
        dropout = 0.15
        emb_dropout = 0.15
        model = DualEfficientViT(
            image_size=args.image_size,
            patch_size=args.patch_size,
            num_classes=1,
            dim=args.dim,
            depth=args.depth,
            heads=args.heads,
            mlp_dim=args.mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout,
            version=args.version,
            weight=args.weight,
            freeze=args.freeze
        )
        
        args_txt = "ver_{}_weight_{}_imgsize_{}_lr_{}_patchsize_{}_es_{}_loss_{}_freeze_{}".format(args.version, args.weight, args.image_size, args.lr, args.patch_size, args.es_metric, args.loss, args.freeze)
        criterion = [args.loss]
        if args.gamma:
            args_txt += "gamma_{}".format(args.gamma)
            criterion.append(args.gamma)
        
        train_dual_stream(model, criterion_name=criterion, train_dir=args.train_dir, val_dir=args.val_dir, image_size=args.image_size, lr=args.lr,\
                           batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, resume=args.resume, epochs=args.n_epochs, eval_per_iters=args.eval_per_iters,\
                           adj_brightness=adj_brightness, adj_contrast=adj_contrast, es_metric=args.es_metric, es_patience=args.es_patience, model_name="dual-efficient-vit", args_txt=args_txt)
        
    elif model == "dual_efficient_vit_v2":
        from module.train_torch import train_dual_stream
        from model.vision_transformer.dual_efficient_vit_v2 import DualEfficientViTV2
        
        dropout = 0.15
        emb_dropout = 0.15
        model = DualEfficientViTV2(
            image_size=args.image_size,
            patch_size=args.patch_size,
            num_classes=1,
            dim=args.dim,
            depth=args.depth,
            heads=args.heads,
            mlp_dim=args.mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout,
            version=args.version,
            weight=args.weight,
            freeze=args.freeze
        )
        
        args_txt = "ver_{}_weight_{}_imgsize_{}_lr_{}_patchsize_{}_es_{}_loss_{}_freeze_{}".format(args.version, args.weight, args.image_size, args.lr, args.patch_size, args.es_metric, args.loss, args.freeze)
        criterion = [args.loss]
        if args.gamma:
            args_txt += "gamma_{}".format(args.gamma)
            criterion.append(args.gamma)
        
        train_dual_stream(model, criterion_name=criterion, train_dir=args.train_dir, val_dir=args.val_dir, image_size=args.image_size, lr=args.lr,\
                           batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, resume=args.resume, epochs=args.n_epochs, eval_per_iters=args.eval_per_iters,\
                           adj_brightness=adj_brightness, adj_contrast=adj_contrast, es_metric=args.es_metric, es_patience=args.es_patience, model_name="dual-efficient-vit", args_txt=args_txt)