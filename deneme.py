from model.scraper import TwitterUserScraper
from model.model import Predictor

df = TwitterUserScraper("gorkem").scrape_tweets()
df["pred"] = df[["text"]].apply(lambda x: Predictor(x["text"]).get_prediction(), axis=1)
print(df["pred"].mean())
