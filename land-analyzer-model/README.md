# Dataset creator

- Run dataset generation:
Linux: `/bin/python3 ./dscreator.py --src-root ./data/sources/dataset/ --dst-root ./data/outputs/dataset/`
Windows: `python.exe .\dscreator.py --src-root .\data\sources\train\ --dst-root .\data\outputs\train\`

# Model trainer

- Run training:
Linux: `/bin/python3 ./trainer.py --config-file ./configs/cfg.yaml --save-step 1 --use-tensorboard 1 --eval-step 1`
Windows: `python.exe .\trainer.py --config-file .\configs\cfg.yaml --save-step 1 --use-tensorboard 1 --eval-step 1`

- Run tensorboard to see learning progress:
Linux: `tensorboard --logdir=./outputs/test1/tf_logs/`
Windows: `tensorboard.exe --logdir .\outputs\test\tf_logs\`

- Export:
Linux: `CUDA_VISIBLE_DEVICES=0 python3 exporter.py --config-file ./outputs/dataset/cfg.yaml --batch-size 1 --target ti`
Windows: `python.exe exporter.py --config-file .\outputs\test\cfg.yaml --batch-size 1 --target ti`

# Model infer

- Run single image infer
Linux: `/bin/python3 ./inferer.py --cfg ./outputs/test_jaccard_2/cfg.yaml --i ./data/infer/test11.png --o ./data/infer/test11_result.png`
Windows: `python.exe .\inferer.py --cfg .\outputs\test\cfg.yaml --i .\data\infer\test.png --o .\data\infer\test_result.png`