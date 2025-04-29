srun --pty --account gax@h100 -C h100 --nodes=1 --ntasks-per-node=1 --cpus-per-task=12 --gres=gpu:1 --time=20:00:00 --hint=nomultithread bash
