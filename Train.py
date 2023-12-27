import argparse
import parses.parses as parses
import train.trainer as trainer


config,nouse=parses.get_config()
train_net=trainer.Trainer(config)
train_net.run()