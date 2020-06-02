import os
import json

import responder

from node_ranking import NodeScorer


env = os.environ
DEBUG = env['DEBUG'] in ['1', 'True', 'true']
LIBRARY = env['LIBRARY']
ALGO = env['ALGO']

cur_dir = os.path.dirname(__file__)
with open(os.path.join(cur_dir, 'node_ranking.json')) as fp:
    CONFIG = json.load(fp)

api = responder.API(debug=DEBUG)
scorer = NodeScorer(
    library=LIBRARY,
    algo=ALGO,
    **CONFIG
)


@api.route("/")
async def score(req, resp):
    body = await req.text
    json_body = json.loads(body)

    resp.media = scorer.score(json_body)


if __name__ == "__main__":
    api.run()