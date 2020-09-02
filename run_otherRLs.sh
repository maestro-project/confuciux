cd src/Other_RLs
python main.py --outdir outdir --model_def vgg16 --fitness latency --cstr area --platform cloud --epochs 100 --df shi --alg PPO2
cd ../../