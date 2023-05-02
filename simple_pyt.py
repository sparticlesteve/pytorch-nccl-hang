import os

import torch
import torch.distributed as dist

class simple_model(torch.nn.Module):
  def __init__(self):
    super(simple_model, self).__init__()
    self.conv0 = torch.nn.Conv2d(4, 512, kernel_size=3, padding=1)
    self.conv1 = torch.nn.Conv2d(512, 1024, kernel_size=3, padding=1)
    self.conv2 = torch.nn.Conv2d(1024, 1024, kernel_size=3, padding=1)

  def forward(self, x):
    return self.conv2(self.conv1(self.conv0(x)))

class simple_dataset(torch.utils.data.Dataset):
  def __init__(self, length):
    super(simple_dataset).__init__()
    self.len = length
  def __getitem__(self, idx):
    return torch.empty(4, 256, 256)
  def __len__(self):
    return self.len


def main():
  torch.backends.cudnn.benchmark=True

  rank = int(os.environ.get("RANK"))
  local_rank = int(os.environ.get("LOCAL_RANK"))
  world_size = int(os.environ.get("WORLD_SIZE"))

  device = torch.device("cuda", local_rank)
  torch.cuda.set_device(device)

  dist.init_process_group("nccl", rank=rank, world_size=world_size)

  model = simple_model().to(device)
  opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
  model = torch.nn.parallel.DistributedDataParallel(model)

  # NOTE: num_workers = 1 will hang, num_workers = 0 will run correctly
  loader = torch.utils.data.DataLoader(simple_dataset(10), batch_size=1, num_workers=1)

  for i, x in enumerate(loader):
    print(f"RANK {rank}: running step {i+1}/{len(loader)}")
    x = x.to(device)
    x = model(x)
    loss = x.sum()
    opt.zero_grad()
    loss.backward()
    opt.step()

  dist.barrier()
  torch.cuda.synchronize()
  print(f"RANK {rank}: COMPLETE!")

if __name__ == "__main__":
  main()
