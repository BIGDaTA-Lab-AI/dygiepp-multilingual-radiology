import json
import jsonlines
import random
import fire


def one_sentence(input_files=["train"], output_file="train", no_of_samples=600):
    data = {}
    for file_name in input_files:
        with open('data/radgraph_1/' + file_name + '.json') as file:
            data = json.load(file)

    dygie_reports = []
    report_names = list(data.keys())
    random.shuffle(report_names)
    count = 0
    for report_name in report_names:
        report = data[report_name]
        tokenized_report = report["text"].split()

        ner = [[]]
        relations = [[]]
        if "entities" in report:
            entities = report["entities"]
        else:
            entities = report["labeler_1"]["entities"]

        for i in entities:
            entity = entities[i]
            ner[0].append([
                entity["start_ix"],
                entity["end_ix"],
                entity["label"]
            ])
            entity_relations = [[
                entity["start_ix"],
                entity["end_ix"],
                entities[relation[1]]["start_ix"],
                entities[relation[1]]["end_ix"],
                relation[0]
            ] for relation in entity["relations"]]
            relations[0] = relations[0] + entity_relations

        dygie_reports.append({
            "doc_key": report_name,
            "sentences": [tokenized_report],
            "ner": ner,
            "relations": relations
        })
        count += 1
        if count == no_of_samples:
            break

    with jsonlines.open('data/' + output_file + '.json', 'w') as writer:
        writer.write_all(dygie_reports)
    print(str(count) + ' samples converted. ' + str(input_files) + ' -> ' + output_file)


def multiple_sentences(input_files=["train"], output_file="train", no_of_samples=600):
    data = {}
    for file_name in input_files:
        with open('data/radgraph_1/' + file_name + '.json') as file:
            data = json.load(file)

    dygie_reports = []
    report_names = list(data.keys())
    random.shuffle(report_names)
    count = 0
    for report_name in report_names:
        report = data[report_name]
        sentences = report["text"].split('.')
        tokenized_sentences = [(sentence.split() + ['.']) for sentence in sentences]
        if len(tokenized_sentences[-1]) == 0 or tokenized_sentences[-1] == ['.']:
            tokenized_sentences.pop()

        tokens_count = [len(sentence) for sentence in tokenized_sentences]
        ner = [[] for _ in tokenized_sentences]
        relations = [[] for _ in tokenized_sentences]
        if "entities" in report:
            entities = report["entities"]
        else:
            entities = report["labeler_1"]["entities"]

        sentence_index = 0
        for i in entities:
            entity = entities[i]
            while entity["start_ix"] >= sum(tokens_count[:sentence_index+1]):
                sentence_index += 1
            start_ix = entity["start_ix"]
            end_ix = entity["end_ix"]
            ner[sentence_index].append([
                start_ix,
                end_ix,
                entity["label"]
            ])
            entity_relations = [[
                start_ix,
                end_ix,
                entities[relation[1]]["start_ix"],
                entities[relation[1]]["end_ix"],
                relation[0]
            ] for relation in entity["relations"]]
            relations[sentence_index] = relations[sentence_index] + entity_relations

        dygie_reports.append({
            "doc_key": report_name,
            "sentences": tokenized_sentences,
            "ner": ner,
            "relations": relations
        })
        count += 1
        if count == no_of_samples:
            break

    with jsonlines.open('data/' + output_file + '.json', 'w') as writer:
        writer.write_all(dygie_reports)
    print(str(count) + ' samples converted')


if __name__ == "__main__":
    fire.Fire()