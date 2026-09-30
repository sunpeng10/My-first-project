"""
中文情感分类 Baseline — CLI 入口

用法:
    python main.py train          # 训练模型
    python main.py eval           # 评估测试集
    python main.py predict        # 交互式预测 或 --text "xxx"
    python main.py predict --text "这家酒店不错"

依赖:
    src.trainer  — 训练 pipeline
    src.evaluate — 评估 pipeline
    src.inference — 预测接口
"""
import argparse
import logging
import sys

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)


def main():
    parser = argparse.ArgumentParser(
        description="中文情感分类 Baseline（RoBERTa + Captum）",
    )
    sub = parser.add_subparsers(dest="command", help="可用命令")

    # ---- train ----
    sub.add_parser("train", help="训练模型")

    # ---- eval ----
    sub.add_parser("eval", help="在测试集上评估已训练模型")

    # ---- predict ----
    pred_parser = sub.add_parser("predict", help="情感预测")
    pred_parser.add_argument(
        "--text", type=str, default=None,
        help="待预测文本（不传则进入交互模式）",
    )

    args = parser.parse_args()

    if args.command == "train":
        from src.trainer import train
        trainer, model, tokenizer = train()
        print(f"\n训练完成！模型保存至: {trainer.args.output_dir}")

    elif args.command == "eval":
        from src.evaluate import evaluate
        evaluate()

    elif args.command == "predict":
        from src.inference import predict, predict_batch

        if args.text:
            # 单次预测
            r = predict(args.text)
            print(f"\n 文本: {r['text']}")
            print(f" 情感: {r['label']}  置信度: {r['score']:.2%}")
        else:
            # 交互模式
            print("交互预测模式（输入 quit 退出）")
            print("-" * 50)
            while True:
                try:
                    text = input("\n请输入文本: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n退出。")
                    break
                if text.lower() in ("quit", "exit", "q"):
                    print("退出。")
                    break
                if not text:
                    continue
                r = predict(text)
                print(f" → {r['label']} (置信度: {r['score']:.2%})")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
