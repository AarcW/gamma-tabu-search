from train_tabu import train_model_tabu
import argparse
import os
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fitness1', type=str, default="latency")
    parser.add_argument('--fitness2', type=str, default="energy")
    parser.add_argument('--num_pop', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--outdir', type=str, default="outdir")
    parser.add_argument('--num_pe', type=int, default=1024)
    parser.add_argument('--l1_size', type=int, default=-1)
    parser.add_argument('--l2_size', type=int, default=-1)
    parser.add_argument('--area_budget', type=float, default=-1)

    # ✅ Add missing arguments back
    parser.add_argument('--NocBW', type=int, default=-1, help='Network-on-Chip BW')
    parser.add_argument('--model', type=str, default="vgg16", help='Model to run')
    parser.add_argument('--singlelayer', type=int, default=0, help='The layer index to optimize')

    opt = parser.parse_args()

    opt = parser.parse_args()

    m_file_path = "../../data/model/"
    m_file = os.path.join(m_file_path, opt.model + ".csv")
    df = pd.read_csv(m_file)
    model_defs = df.to_numpy()

    chkpt_file = os.path.join(opt.outdir, "result_c.plt")

    train_model_tabu(model_defs, input_arg=opt, chkpt_file=chkpt_file)
