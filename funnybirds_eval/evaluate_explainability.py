import argparse
import random

import torch
from captum.attr import InputXGradient, IntegratedGradients
from evaluation_protocols import (
    accuracy_protocol,
    background_independence_protocol,
    controlled_synthetic_data_check_protocol,
    deletion_check_protocol,
    distractibility_protocol,
    preservation_check_protocol,
    single_deletion_protocol,
    target_sensitivity_protocol,
)
from explainers.explainer_wrapper import (
    CaptumAttributionExplainer,
    ViTCheferLRPExplainer,
    ViTRolloutExplainer,
)
from models.model_wrapper import StandardModel, ViTModel
from models.resnet import resnet50
from models.vgg import vgg16
from models.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP
from models.ViT.ViT_new import vit_base_patch16_224


parser = argparse.ArgumentParser(description="FunnyBirds - Explanation Evaluation")
parser.add_argument(
    "--data", metavar="DIR", required=True, help="path to dataset (default: imagenet)"
)
parser.add_argument(
    "--model",
    required=True,
    choices=["resnet50", "vgg16", "vit_b_16"],
    help="model architecture",
)
parser.add_argument(
    "--explainer",
    required=True,
    choices=[
        "IntegratedGradients",
        "InputXGradient",
        "Rollout",
        "CheferLRP",
        "CustomExplainer",
        "CIAO",
    ],
    help="explainer",
)
parser.add_argument(
    "--checkpoint_name",
    type=str,
    required=False,
    default=None,
    help="checkpoint name (including dir)",
)

# CIAO-specific arguments
parser.add_argument(
    "--ciao_method",
    default="lookahead",
    choices=["lookahead", "mcts", "mcgs", "ucb", "potential", "pure_monte_carlo", "beam_search"],
    help="CIAO search algorithm (default: lookahead)",
)
parser.add_argument(
    "--ciao_segmentation",
    default="hex",
    choices=["hex", "square"],
    help="CIAO segmentation type (default: hex)",
)
parser.add_argument(
    "--ciao_hex_radius",
    default=8,
    type=int,
    help="Hex circumradius in pixels for hex segmentation (default: 8)",
)
parser.add_argument(
    "--ciao_square_size",
    default=8,
    type=int,
    help="Square patch edge length in pixels for square segmentation (default: 8)",
)
parser.add_argument(
    "--ciao_max_regions",
    default=5,
    type=int,
    help="Maximum number of regions CIAO will find (default: 5)",
)
parser.add_argument(
    "--ciao_desired_length",
    default=30,
    type=int,
    help="Target number of segments per region (default: 30)",
)
parser.add_argument(
    "--ciao_batch_size",
    default=16,
    type=int,
    help="Internal batch size for CIAO forward passes (default: 16)",
)
parser.add_argument(
    "--ciao_step_budget",
    default=64,
    type=int,
    help="Step budget for UCB/potential (default: 64)",
)
parser.add_argument(
    "--ciao_num_evals",
    default=6400,
    type=int,
    help="Eval budget for MCTS/MCGS (default: 6400)",
)
parser.add_argument(
    "--ciao_num_rollouts",
    default=64,
    type=int,
    help="Rollouts per iteration for MCTS/MCGS (default: 64)",
)
parser.add_argument(
    "--ciao_lookahead_distance",
    default=2,
    type=int,
    help="Lookahead distance (default: 2)",
)
parser.add_argument(
    "--ciao_sigma",
    default=1,
    type=int,
    choices=[-1, 0, 1],
    help="Seed selection sign: 1=positive evidence, -1=negative evidence, 0=auto (default: 1)",
)

parser.add_argument(
    "--ciao_mlflow_tracking_uri",
    default=None,
    type=str,
    help="MLflow tracking URI (omit to disable logging)",
)
parser.add_argument(
    "--ciao_mlflow_experiment",
    default="ciao-funnybirds",
    type=str,
    help="MLflow experiment name (default: ciao-funnybirds)",
)
parser.add_argument(
    "--ciao_mlflow_run_name",
    default=None,
    type=str,
    help="MLflow parent run name",
)

parser.add_argument("--gpu", default=0, type=int, help="GPU id to use, or -1 for CPU.")
parser.add_argument("--seed", default=0, type=int, help="seed")
parser.add_argument(
    "--batch_size",
    default=32,
    type=int,
    help="batch size for protocols that do not require custom BS such as accuracy",
)
parser.add_argument(
    "--nr_itrs",
    default=2501,
    type=int,
    help="batch size for protocols that do not require custom BS such as accuracy",
)

parser.add_argument(
    "--accuracy", default=False, action="store_true", help="compute accuracy"
)
parser.add_argument(
    "--controlled_synthetic_data_check",
    default=False,
    action="store_true",
    help="compute controlled synthetic data check",
)
parser.add_argument(
    "--single_deletion",
    default=False,
    action="store_true",
    help="compute single deletion",
)
parser.add_argument(
    "--preservation_check",
    default=False,
    action="store_true",
    help="compute preservation check",
)
parser.add_argument(
    "--deletion_check",
    default=False,
    action="store_true",
    help="compute deletion check",
)
parser.add_argument(
    "--target_sensitivity",
    default=False,
    action="store_true",
    help="compute target sensitivity",
)
parser.add_argument(
    "--distractibility",
    default=False,
    action="store_true",
    help="compute distractibility",
)
parser.add_argument(
    "--background_independence",
    default=False,
    action="store_true",
    help="compute background dependence",
)


def main():
    args = parser.parse_args()
    device = "cpu" if args.gpu < 0 else f"cuda:{args.gpu}"

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    mlflow_enabled = args.explainer == "CIAO" and args.ciao_mlflow_tracking_uri is not None
    if mlflow_enabled:
        import mlflow
        mlflow.set_tracking_uri(args.ciao_mlflow_tracking_uri)
        mlflow.set_experiment(args.ciao_mlflow_experiment)
        parent_run = mlflow.start_run(run_name=args.ciao_mlflow_run_name)
        mlflow.log_params({
            "method": args.ciao_method,
            "segmentation": args.ciao_segmentation,
            "hex_radius": args.ciao_hex_radius,
            "max_regions": args.ciao_max_regions,
            "desired_length": args.ciao_desired_length,
            "ciao_batch_size": args.ciao_batch_size,
            "nr_itrs": args.nr_itrs,
        })

    # create model
    if args.model == "resnet50":
        model = resnet50(num_classes=50)
        model = StandardModel(model)
    elif args.model == "vgg16":
        model = vgg16(num_classes=50)
        model = StandardModel(model)
    elif args.model == "vit_b_16":
        if args.explainer == "CheferLRP":
            model = vit_LRP(num_classes=50)
        else:
            model = vit_base_patch16_224(num_classes=50)
        model = ViTModel(model)
    else:
        print("Model not implemented")

    if args.checkpoint_name:
        model.load_state_dict(
            torch.load(args.checkpoint_name, map_location=torch.device("cpu"))[
                "state_dict"
            ]
        )
    model = model.to(device)
    model.eval()

    # create explainer
    if args.explainer == "InputXGradient":
        explainer = InputXGradient(model)
        explainer = CaptumAttributionExplainer(explainer)
    elif args.explainer == "IntegratedGradients":
        explainer = IntegratedGradients(model)
        baseline = torch.zeros((1, 3, 256, 256)).to(device)
        explainer = CaptumAttributionExplainer(explainer, baseline=baseline)
    elif args.explainer == "Rollout":
        explainer = ViTRolloutExplainer(model)
    elif args.explainer == "CheferLRP":
        explainer = ViTCheferLRPExplainer(model)
    elif args.explainer == "CustomExplainer":
        ...
    elif args.explainer == "CIAO":
        from ciao.explainer.explanation_methods import (
            make_beam_search_method,
            make_lookahead_method,
            make_mcgs_method,
            make_mcts_method,
            make_potential_method,
            make_pure_monte_carlo_method,
            make_ucb_method,
        )
        from explainers.ciao_explainer import CIAOFunnyBirdsExplainer

        _method_map = {
            "lookahead": make_lookahead_method(args.ciao_lookahead_distance),
            "mcts": make_mcts_method(args.ciao_num_evals, args.ciao_num_rollouts),
            "mcgs": make_mcgs_method(args.ciao_num_evals, args.ciao_num_rollouts),
            "ucb": make_ucb_method(step_budget=args.ciao_step_budget, batch_size=args.ciao_batch_size),
            "potential": make_potential_method(step_budget=args.ciao_step_budget),
            "pure_monte_carlo": make_pure_monte_carlo_method(num_evals=args.ciao_num_evals),
            "beam_search": make_beam_search_method(),
        }
        class_names = [str(i) for i in range(50)]
        explainer = CIAOFunnyBirdsExplainer(
            model=model,
            class_names=class_names,
            method=_method_map[args.ciao_method],
            segmentation=args.ciao_segmentation,
            hex_radius=args.ciao_hex_radius,
            square_size=args.ciao_square_size,
            max_regions=args.ciao_max_regions,
            desired_length=args.ciao_desired_length,
            ciao_batch_size=args.ciao_batch_size,
            sigma=None if args.ciao_sigma == 0 else args.ciao_sigma,
            mlflow_enabled=mlflow_enabled,
        )
    else:
        print("Explainer not implemented")

    accuracy, csdc, pc, dc, distractibility, sd, ts = -1, -1, -1, -1, -1, -1, -1

    if args.accuracy:
        print("Computing accuracy...")
        accuracy = accuracy_protocol(model, args)
        accuracy = round(accuracy, 5)

    if args.controlled_synthetic_data_check:
        print("Computing controlled synthetic data check...")
        csdc = controlled_synthetic_data_check_protocol(model, explainer, args)

    if args.target_sensitivity:
        print("Computing target sensitivity...")
        ts = target_sensitivity_protocol(model, explainer, args)
        ts = round(ts, 5)

    if args.single_deletion:
        print("Computing single deletion...")
        sd = single_deletion_protocol(model, explainer, args)
        sd = round(sd, 5)

    if args.preservation_check:
        print("Computing preservation check...")
        pc = preservation_check_protocol(model, explainer, args)

    if args.deletion_check:
        print("Computing deletion check...")
        dc = deletion_check_protocol(model, explainer, args)

    if args.distractibility:
        print("Computing distractibility...")
        distractibility = distractibility_protocol(model, explainer, args)

    if args.background_independence:
        print("Computing background independence...")
        background_independence = background_independence_protocol(model, args)
        background_independence = round(background_independence, 5)

    # select completeness and distractability thresholds such that they maximize the sum of both
    max_score = 0
    best_threshold = -1
    for threshold in csdc:
        max_score_tmp = (
            csdc[threshold] / 3.0
            + pc[threshold] / 3.0
            + dc[threshold] / 3.0
            + distractibility[threshold]
        )
        if max_score_tmp > max_score:
            max_score = max_score_tmp
            best_threshold = threshold

    print("FINAL RESULTS:")
    print("Accuracy, CSDC, PC, DC, Distractability, Background independence, SD, TS")
    print(
        f"{accuracy}\t{round(csdc[best_threshold], 5)}\t{round(pc[best_threshold], 5)}\t{round(dc[best_threshold], 5)}\t{round(distractibility[best_threshold], 5)}\t{background_independence}\t{sd}\t{ts}"
    )
    print("Best threshold:", best_threshold)

    if mlflow_enabled:
        mlflow.log_metrics({
            "accuracy": accuracy,
            "csdc": round(csdc[best_threshold], 5),
            "pc": round(pc[best_threshold], 5),
            "dc": round(dc[best_threshold], 5),
            "distractibility": round(distractibility[best_threshold], 5),
            "background_independence": background_independence,
            "sd": sd,
            "ts": ts,
            "best_threshold": best_threshold,
        })
        mlflow.end_run()


if __name__ == "__main__":
    main()
