from flask import Flask, render_template, request, jsonify
from be.Cold_model.cold_associatedrules_rec import associated_rules, generate_tweets
from be.Hot_model.hot_associatedrules_rec import associated_rules as hot_associated_rules, generate_tweets as hot_generate_tweets
from be.General_process.image_rec import TextSimilarityFinder
from be.Cold_model.cold_opti import TweetGenerator
import os

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/handle_selections', methods=['POST'])
def handle_request():
    if request.method == 'POST':
        data = request.get_json()

        button_states = data.get('buttonStates')
        selected_values = data.get('selectedValues')

        print('Button States:', button_states)
        print('Selected Values:', selected_values)

        if len(selected_values) <= 10:
            if selected_values and len(selected_values) > 2:
                category = selected_values[0]
                selected_values = selected_values[1:-1]

                print('Updated Selected Values:', selected_values)

                ruleOutput = associated_rules(selected_values)
                tweetGenerate = [generate_tweets(category)][0]
            else:
                return jsonify({'error': 'Invalid selected_values'})
        else:
            if selected_values and len(selected_values) > 2:
                start_date = selected_values[0]
                end_date = selected_values[1]
                category = selected_values[2]
                selected_values = selected_values[3:-1]

                print('Updated Selected Values:', selected_values)

                ruleOutput = hot_associated_rules(
                    selected_values, start_date, end_date)
                tweetGenerate = [hot_generate_tweets(category)][0]

        response_data = {'ruleOutput': ruleOutput,
                         'tweetGenerate': tweetGenerate, 'Selected Values': selected_values}
        return jsonify(response_data)


@app.route('/handle_user_input', methods=['POST'])
def handle_user_input():
    if request.method == 'POST':
        data = request.get_json()

        file_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "be/Cold_model/cold_data.xlsx")

        # Retrieve user input and features input
        user_input = data.get('userInput')
        categoryInput = data.get('categoryInput')  # Corrected variable name

        tweet_generator = TweetGenerator(file_path)
        generated_tweets = tweet_generator.generate_tweets(
            user_input, categoryInput)

        return jsonify(generated_tweets)


@app.route('/process_picture_input', methods=['POST'])
def process_picture_input():
    data = request.get_json()

    input_text = data.get('input')

    data_path = os.path.join(os.path.dirname(os.path.abspath(
        __file__)), "be/General_process/Processed_Tweets.xlsx")
    similarity_finder = TextSimilarityFinder(data_path)
    most_similar_text, most_similar_imagelabel = similarity_finder.find_most_similar_text(
        input_text)

    if most_similar_text and most_similar_imagelabel:
        imagekey = "Recommended image labels: " + "\n" + most_similar_imagelabel + \
            "\n" + similarity_finder.generate_imagekey(most_similar_imagelabel)
    else:
        imagekey = similarity_finder.generate_imagekey(input_text)

    return jsonify({'result': imagekey})


if __name__ == '__main__':
    app.run(debug=True)
