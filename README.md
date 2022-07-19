### Emotion classification

Machine learning system for classifying emotions.
### Data

Dataset used: GoEmotions that contains curated list of 58k Reddit comments labeled into  27 emotions. Its detailed description can be found at the following [link](https://huggingface.co/datasets/go_emotions).
github: [GoEmotions github](https://github.com/google-research/google-research/tree/master/goemotions)

#### Problems in question

There are three possible ways of approaching the problem, depending on the number of classes (clusters).
For now, no clustering is considered.

### Overall scheme
![overall_scheme](https://user-images.githubusercontent.com/37189954/179788142-9a73f156-0c91-4138-8bdb-d5a9b574917c.png)

### Installation

Dependencies: Python 3.9+
Clone the repository and create virtual environment
```
git clone git@github.com:BartheqVonKoss/emoclass.git
python -m venv env
```
Activate it and install required packages, those are kept in `requirements.txt`
```
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Last step is to create configuration file. This can be done for example by
```
cp config/training_config_template.py config/training_config.py
```
The project will keep track of git-tracked files as well as logs and models.

#### Inference 
There are two modes for inference supported:
- command line interface that takes `--text` (text string with a quote to classify) and `--model` model to infer from, using
 
  ```
  python infer.py
  ```
  
- web-hosted gradio interface
  * run locally and start web interface at `localhost:1212`
   
  ```
  python app.py
  ```
  
  
  * running on spaces by HuggingFace

All inference modes assume there are present saved models as well as the vocabulary used to fit training dataset.

#### Training

In order to train a model update configuration file `config/training_config.py` and provide with a training dataset. In case of another dataset used, user might need to update `src/dataset/dataset.py` with proper implementation of dataset and emotions.
To change text preprocessing steps edit `preprocess_helper` dictionary in `src/dataset/preprocess_helper.py`.

To start training run
```
python train.py
```
Once optimization is finished, the trained model is ready for running predictions.

#### Docker

It might be the case that the web interface needs to be accessibile on remote machine, from within docker container. If so, install docker on the machine and start service, for example

```
pacman -S docker
systemctl start docker.service
````

Next build container using default `Dockerfile`. This process, if run for the first time may take considerable time ( < 3 min).
```
docker build -t emoclass .
```
Once successful, one can run docker container
```
docker run -it --rm -p 1212:1212 --name gradio_ai emoclass:latest
```
and have the inteface over `localhost:1212` in the web browser.

#### Note
It might be necessary for some os to run the above commands with `sudo` permissions.
#### Miscalleanous
Code should be compatibile with pep8 rules. Some exceptions are handled inside `setup.cfg`.
