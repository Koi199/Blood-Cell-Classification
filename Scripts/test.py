# keep top-level sys.path modifications so imports resolve
import sys
import copy
import os

sys.path.append("C:/repos/Blood-Cell-Classification/Scripts/Validation")
from K_foldValidation import kfold_validate

sys.path.append("C:/repos/Blood-Cell-Classification/Scripts/Logging")

# NOTE: move classifier imports into main() to avoid side-effects at import time

def main():
    # import training code and dataset loader inside main to avoid running on child import
    sys.path.append("C:/repos/Blood-Cell-Classification/Scripts/Classifiers/Grid_Search")
    from Classifier_MCvsNonMC import train, DEFAULT_CONFIG, load_samples, COLLAPSE_MAP

    # prepare config
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["architecture"]    = "resnet34"
    config["data_dir"]        = r"D:\MMA_LabelledData\Sliced"
    config["checkpoint_path"] = r"C:\repos\Blood-Cell-Classification\checkpoints_stage1\resnet34_kfold.pth"
    config["seed"]            = 42

    # ensure checkpoint directory exists
    os.makedirs(os.path.dirname(config["checkpoint_path"]), exist_ok=True)

    # OPTIONAL: set multiprocessing start method (must be inside __main__ guard)
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # start method already set; ignore
        pass

    # run k-fold validation
    aggregate = kfold_validate(
        train_fn        = train,
        config          = config,
        load_fn         = load_samples,
        label_key       = "binary",
        collapse_map    = COLLAPSE_MAP,
        experiment_name = "Stage1_KFold",
        notes           = "resnet34 baseline kfold",
        n_splits        = 5,
        results_csv     = None
    )

    print(f"Final aggregate test_acc: {aggregate.get('test_acc_mean'):.4f} ± {aggregate.get('test_acc_std'):.4f}")

if __name__ == "__main__":
    # Windows: support frozen executables and safe spawn
    from multiprocessing import freeze_support
    freeze_support()
    main()
