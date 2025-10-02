# FedReal

依赖：pip install torch torchvision torchaudio grpcio grpcio-tools

## 生成 gRPC 代码

在项目根目录（包含 proto/ 文件夹）执行：

python -m grpc_tools.protoc -I proto --python_out=proto --grpc_python_out=proto proto/fed.proto

将在 proto/ 下生成 fed_pb2.py 与 fed_pb2_grpc.py

注意重新生成后要修改fed_pb2_grpc.py 

把 import fed_pb2 as fed__pb2 改成 from . import fed_pb2 as fed__pb2

## 启动示例

在根目录下运行launch.py，具体命令参照scripts下的README.md

### Server
python -m server.server_main \
  --bind "0.0.0.0:50052" \
  --data_root "./dataset" \
  --dataset_name "NWPU-RESISC45" \
  --num_classes 45 \
  --num_clients 19 \
  --rounds 10 \
  --local_epochs 3 \
  --batch_size 32 \
  --lr 0.005 \
  --momentum 0.9 \
  --sample_fraction 1.0 \
  --seed 42 \
  --model_name "resnet18" \
  --feature_dim 512 \
  --max_message_mb 256 \
  --encoder_ratio 0.2 \
  --algorithm "FedEXT" \
  --device "cpu" \
  --run_dir "./runs/NWPU-RESISC45/FedEXT/$(date '+%Y%m%d_%H%M%S')" \
  --feature_batch_size 128 \
  --tail_batch_size 64 \
  --tail_epochs 2 \
  --tail_lr 0.01 \
  --tail_momentum 0.9 \
  --tail_weight_decay 1e-4 \
  --tail_device "cpu" \
  --tail_model_name "resnet34" \
  --feature_keep_spatial \
  2>&1 | tee logs/server_nwpu_$(date '+%Y%m%d_%H%M%S').log

### Client
python -m client.client_main \
  --server "192.168.202.197:50052" \
  --client_name "$(hostname -I | awk '{print $1}')" \
  --data_root "./dataset" \
  --dataset_name "NWPU-RESISC45" \
  --num_classes 45 \
  --num_clients 19 \
  --batch_size 64 \
  --seed 42 \
  --feature_dim 512 \
  --encoder_ratio 0.2 \
  --algorithm "FedEXT" \
  --max_message_mb 256 \
  --device "cpu" \
  2>&1 | tee logs/client_nwpu_$(hostname -I | awk '{print $1}').log

