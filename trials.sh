python main.py --mode train --model_path model.bin --hidden_size_dec 60 --hidden_size_enc 90  > test-a-output.txt
python main.py --mode train --model_path model.bin --hidden_size_dec 90 --hidden_size_enc 60  > test-b-output.txt
python main.py --mode train --model_path model.bin --hidden_size_dec 40 --hidden_size_enc 80 --attention ADDITIVE > test-c-output.txt

python main.py --mode train --model_path model.bin --dropout 0.2 > test-d-output.txt
python main.py --mode train --model_path model.bin --dropout 0.4 > test-e-output.txt

python main.py --mode train --model_path model.bin --patience 10 --lr_decay 0.6 > test-f-output.txt
python main.py --mode train --model_path model.bin --patience 3 --lr_decay 0.4 > test-g-output.txt

python main.py --mode train --model_path model.bin --batch 100 --max_epoch 30 > test-h-output.txt
