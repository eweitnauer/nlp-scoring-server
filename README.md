# NLP Scoring Server

This repository combines several algorithms from the natural language processing field (NLP) to determine the similarity of sentences.

The work is funded by an IES grant of Dr. Michael N. Jones and Dr. Jeffrey D. Karpicke. The encoder and classifier components are based on [this work](https://github.com/FTAsr/STS) from Dr. Jones' group. The server part is inspired by the implementation of [Anirudh](https://github.com/anirudhchellani) in Dr. Karpicke's group.

Please see the [SETUP.md](https://github.com/eweitnauer/nlp-scoring-server/blob/master/SETUP.md) file about how to install this project.

## API Specs v1.2

### Score Requests

In order to get a score for the similarity of a response string to a target string, send a get request to `/api/score`. Provide the following query parameters:

| Parameter | Required | Use Case | Type |
| :---: | :---: | :--- | :---: |
| api_key | required | authenticates the api call (authentication can be disabled in the `server.py` file) | string |
| target | required | target/gold answer to compare the student response to | string |
| response | required | response supplied by the student/user | string  |
| models | required | comma-separated list of encoder models. See below for available models. | string |
| classifier | optional | name of a trained classifier (must fit the model-combination). | string |

Here are some example API calls:

- `/api/score?target=autumn&response=fall&models=quickscore`
- `/api/score?target=plants need water and sunlight to grow&response=plants need the sun&models=infersent`
- `/api/score?target=plants need water and sunlight to grow&response=plants need the sun&models=bow,feature_based&classifier=bow_fb-sick`


The server currently supports the following encoder models:

| Model Name | Description |
| :---: | :--- |
| infersent | embeds a whole sentence in a 4096 dim. vector; more details [here](https://github.com/facebookresearch/InferSent) |
| bow | "bag of words"; adds up the 300 dim. vector embeddings of all words in the sentence using [these pretrained embeddings](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) |
| quickscore | calculates the proportion of words in the target that are included in the response, using spell correction, stemming and synonyms |
| feature_based | combines the quickscore features with several length and token features of both target and response |

Currently, we have the following pre-trained classifiers:

| Classifier Name | Encoder Combination | Target Dataset |
| :---: | :---: | :---: |
| bow_fb-sick | bow, feature_based | SICK |
| infersent-sick | infersent | SICK |

### Score Responses

| Parameter | Optional | Description | Type |
| :---: | :---: | :--- | :---: |
| name  | always | used to provide a string which reflect the name of the api, i.e. Automated Scoring | string |  
| version | always | value representing version of the api called | semantic version number |
| errors | optional | a list of errors that occurred | array of strings |
| score | optional | the similarity score between target and response | number between 0 and 1 |
| models | optional | the models that were used | array of strings |
| classifier | optional | the classifier that was used | string |

### Score Paragraph Requests

In order to get a score matrix that compares several targets (idea units) with a student response, send a `GET` or `POST` request to `/api/score-par`. The student response will be split into sentences. Each response sentence will be compared to each idea unit using the same method as in the Score Requests. Provide the following query parameters:

| Parameter | Required | Use Case | Type |
| :---: | :---: | :--- | :---: |
| api_key | required | authenticates the api call (authentication can be disabled in the `server.py` file) | string |
| targets | required | JSON string of array of idea units to compare the student response to | string |
| response | required | response supplied by the student/user, will be split into sentences | string  |
| models | required | comma-separated list of encoder models. See below for available models. | string |
| classifier | optional | name of a trained classifier (must fit the model-combination). | string |

Here are some example API calls:

- `/api/score-par?targets=["plants need water to grow", "plants need sunlight to grow"]&response=plants grow in the sun. plants are green.&models=bow,feature_based&classifier=bow_fb-sick`

See the previous section for supported encoder models and classifiers.

### Score Responses

| Parameter | Optional | Description | Type |
| :---: | :---: | :--- | :---: |
| name  | always | used to provide a string which reflect the name of the api, i.e. Automated Scoring | string |  
| version | always | value representing version of the api called | semantic version number |
| errors | optional | a list of errors that occurred | array of strings |
| sentences | optional | student response split into sentences | array of strings |
| scores | optional | array of score arrays; the 1st score array contains the similarity scores of the first idea unit to all response sentences | array of arrays of scores between 0 and 1 |
| models | optional | the models that were used | array of strings |
| classifier | optional | the classifier that was used | string |
