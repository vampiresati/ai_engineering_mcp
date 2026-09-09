from sentence_transformers import CrossEncoder
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

# query = "How many people live in Berlin?"
# query = "How many visitor visit in berlin yearly in Berlin?"
query="how many sport and fitness wre registered in 2013 "
passages = [
    "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
    "Berlin has a yearly total of about 135 million day visitors, making it one of the most-visited cities in the European Union.",
    "In 2013 around 600,000 Berliners were registered in one of the more than 2,300 sport and fitness clubs.",
]
value_to_perdict=[(query, passage) for passage in passages]
# # print(value_to_perdict)
# for v in value_to_perdict:
#     print('v')
#     print(v)
# # 2a. Either predict scores pairs of texts
scores = model.predict(value_to_perdict)
print(scores)

ranks = model.rank(query, passages, return_documents=True)

print("Query:", query)
print('rank')
print(ranks)
for rank in ranks:
    print(f"- #{rank['corpus_id']} ({rank['score']:.2f}): {rank['text']}")
