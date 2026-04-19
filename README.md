# Pensieve
Pensieve is a system that generates adaptive bitrate algorithms using reinforcement learning.
http://web.mit.edu/pensieve/

### Prerequisites
- Create a virtual environment with Python 3.8
- This migration was tested around Ubuntu 18.04, Python 3.8, TensorFlow 2.7.0 and TFLearn 0.5.0
- For running `rl_server/rl_server_no_training.py`, the practical minimum is the following pinned set:
```
pip install tensorflow==2.7.0 tflearn==0.5.0 'protobuf<3.21' 'Pillow<10'
```

Notes:
- `setup.py` is kept from the legacy Pensieve workflow and still performs broad system-level installation steps.
- Do not run `python setup.py` on a modern host unless you explicitly want the old Mahimahi / Apache / Selenium setup.
- Online inference does not require a GPU. CPU execution is sufficient.
- Training can run on CPU, but it is significantly slower than GPU training.
- The TensorFlow checkpoint loader in `rl_server_no_training.py` still assumes you start the script from `rl_server/`.
- A working startup sequence on a modern host is:
```
cd rl_server
python rl_server_no_training.py
```

### Training
- To train a new model, put training data in `sim/cooked_traces` and testing data in `sim/cooked_test_traces`, then in `sim/` run `python get_video_sizes.py` and then run
```
python multi_agent.py
```

The reward signal and meta-setting of video can be modified in `multi_agent.py` and `env.py`.
Monitoring the testing curve of rewards, entropy and td loss can be done by launching tensorboard from the terminal as follows:
```
tensorboard --logdir=path/to/results
```
Where path/to/results is in dir `sim`. More details can be found in `sim/README.md`.

### Testing
- To test the trained model in simulated environment, first copy over the model to `test/models` and modify the `NN_MODEL` field of `test/rl_no_training.py` , and then in `test/` run `python get_video_sizes.py` and then run 
```
python rl_no_training.py
```

Similar testing can be performed for buffer-based approach (`bb.py`), mpc (`mpc.py`) and offline-optimal (`dp.cc`) in simulations. More details can be found in `test/README.md`.

### Running experiments over Mahimahi
- To run experiments over mahimahi emulated network, first copy over the trained model to `rl_server/results` and modify the `NN_MODEL` filed of `rl_server/rl_server_no_training.py`, and then in `run_exp/` run
```
python run_all_traces.py
```
This script will run all schemes (buffer-based, rate-based, Festive, BOLA, fastMPC, robustMPC and Pensieve) over all network traces stored in `cooked_traces/`. The results will be saved to `run_exp/results` folder. More details can be found in `run_exp/README.md`.

### Real-world experiments
- To run real-world experiments, first setup a server (`setup.py` automatically installs an apache server and put needed files in `/var/www/html`). Then, copy over the trained model to `rl_server/results` and modify the `NN_MODEL` filed of `rl_server/rl_server_no_training.py`. Next, modify the `url` field in `real_exp/run_video.py` to the server url. Finally, in `real_exp/` run
```
python run_exp.py
```
The results will be saved to `real_exp/results` folder. More details can be found in `real_exp/README.md`.

#### Training and cross-validation (testing) Curve using tensorboard
The RL-model converges after 3 days of continuous training using training data in `sim/cooked_traces` and testing data in `sim/cooked_test_traces`.

TD Loss & Total Reward of `sim/multi-agent.py`: <img src="./sim/cross-validation.png">
