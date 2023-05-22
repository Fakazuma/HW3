import torch
import yaml
from models import trainer

from data.datamodule import DataManager
from txt_logger import TXTLogger
from models.seq2seq_transformer import Seq2SeqTransformer
from data.bpe_tokenizer import BPETokenizer

if __name__ == "__main__":
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = 'cpu'

    # DEVICE = 'cpu'
    print(DEVICE)
    data_config = yaml.load(open("configs/data_config.yaml", 'r'), Loader=yaml.Loader)
    dm = DataManager(data_config, DEVICE)
    train_dataloader, dev_dataloader = dm.prepare_data()

    model_config = yaml.load(open("configs/model_config.yaml", 'r'), Loader=yaml.Loader)

    scheduler_params = {'step_size': model_config['sched_step_size'],
                        'gamma': model_config['sched_gamma']}

    transformer_params = {'nhead': model_config['nhead'],
                          'num_encoder_layers': model_config['num_layers'],
                          'num_decoder_layers': model_config['num_layers'],
                          'dim_feedforward': model_config['dim_feedforward']
                          }

    model = Seq2SeqTransformer(device=DEVICE,
                               emb_size=model_config['embedding_size'],
                               vocab_size=model_config['max_vocab_size'],
                               max_seq_len=model_config['max_seq_len'],
                               target_tokenizer=dm.target_tokenizer,
                               scheduler_params=scheduler_params,
                               transformer_params=transformer_params,
                               )

    logger = TXTLogger('training_logs')
    trainer_cls = trainer.Trainer(model=model, model_config=model_config, logger=logger)

    if model_config['try_one_batch']:
        train_dataloader = [list(train_dataloader)[0]]
        dev_dataloader = [list(dev_dataloader)[0]]

    trainer_cls.train(train_dataloader, dev_dataloader)

#--------------------------------------------------------
    with torch.no_grad():
        with open('Sentences4fun.txt') as f:
            sents = f.read().split('\n')

        test_sents = [dm.source_tokenizer(s) for s in sents]
        test_target = [dm.target_tokenizer(s) for s in ['SOS'] * len(sents)]
        test_ = torch.tensor(test_sents).to(DEVICE)
        test_target_ = torch.tensor(test_target).to(DEVICE)

        predicted_ids_list = model(test_, test_target_)

        predicted, _ = predicted_ids_list
        predicted = predicted.squeeze(-1).detach().cpu().numpy()[:, 1:]

        for i in predicted.astype(int):
            print(dm.target_tokenizer.decode(i))


