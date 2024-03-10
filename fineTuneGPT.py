import pandas as pd
import subprocess

df = pd.read_csv('df_total_sel.csv').sample(n=1500)

prepared_data = df.loc[:,['Title','text']]
prepared_data.rename(columns={'Title':'prompt', 'text':'completion'}, inplace=True)
prepared_data.to_csv('prepared_data.csv',index=False)

### finetunnig completion
# subprocess.run('openai tools fine_tunes.prepare_data --file prepared_data.csv --quiet'.split())
# from openai import OpenAI
# client = OpenAI(api_key = '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')

# training_file  = client.files.create(
#   file=open("prepared_data_prepared.jsonl", "rb"),
#   purpose="fine-tune"
# )
# # Create Fine-Tuning Job
# suffix_name = "books"
# response = client.fine_tuning.jobs.create(
#     training_file=training_file.id,
#     #validation_file=validation_file.id,
#     model="davinci-002",
#     suffix=suffix_name,
# )

# new_prompt = "resume the coments from The Infinity Gauntlet in positive or negative"
# answer = client.completions.create(
#   model='ft:davinci-002:main:books:91GzZnS9',
#   prompt=new_prompt,
#   max_tokens= 50
# )
# print(answer.choices[0].text)




################  finetunign gpt 3.5
import json
def convert_to_gpt35_format(dataset):
    fine_tuning_data = []
    for _, row in dataset.iterrows():
        json_response = '{"description": "' + str(row['description']) + '", "authors": "' + str(row['authors']) +  '", "publisher" : "' + str(row['publisher']) +  '", "text" : "' + str(row['text']) +  '"}'
        fine_tuning_data.append({
            "messages": [
                {"role": "system", "content" : 'You are a content specialist at a certain book publisher. Will answer specific questions about book titles.'},
                {"role": "user", "content": f'Provide me information about {row["Title"]}'},
                {"role": "assistant", "content": json_response}
            ]
        })
    return fine_tuning_data

converted_data = convert_to_gpt35_format(df)



def write_to_jsonl(data, file_path):
    with open(file_path, 'w') as file:
        for entry in data:
            json.dump(entry, file)
            file.write('\n')
            
training_file_name = "train.jsonl"
write_to_jsonl(converted_data, training_file_name)

from openai import OpenAI
client = OpenAI(api_key="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")

# Upload Training and Validation Files
training_file = client.files.create(
    file=open(training_file_name, "rb"), purpose="fine-tune"
)

# Create Fine-Tuning Job
suffix_name = "books"
response = client.fine_tuning.jobs.create(
    training_file=training_file.id,
    #validation_file=validation_file.id,
    model="gpt-3.5-turbo",
    suffix=suffix_name,
)



answer = client.chat.completions.create(
  model='ft:gpt-3.5-turbo-0125:main:yt-tutorial:91Hs3R51',
  messages=[
    {"role": "system", "content": "You are a content specialist at a certain book publisher. Will answer specific questions about book titles."},
    {"role": "user", "content": "Provide me information about 'Now We Are Six'"}
  ],
  
  max_tokens= 50
)
print(answer.choices[0].message.content)