import datetime
import pandas as pd
import snscrape.modules.twitter as sntwitter
import arrow

FROM_DATE = arrow.now().shift(years=-1).format('YYYY-MM-DD')
END_DATE = datetime.date.today()
MAX_RESULTS = 1200

# Using TwitterSearchScraper to scrape data and append tweets to list


class TwitterUserScraper:
    def __init__(self, user_name, from_date=FROM_DATE, end_date=END_DATE, max_results=MAX_RESULTS):
        self.user_name = user_name
        self.from_date = from_date
        self.end_date = end_date
        self.max_results = max_results
        self.tweets_list = []
        self.tweets_df = self.scrape_tweets()
        self.get_tweets_df()

    def scrape_tweets(self):
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(
                f'(from:@{self.user_name}) since:{self.from_date} until:{self.end_date} -filter:replies -filter:replies lang:tr').get_items()):

            if i > MAX_RESULTS:
                break
            self.tweets_list.append(
                [tweet.date, tweet.id, tweet.content, tweet.user.username, tweet.quotedTweet, tweet.retweetedTweet,
                 tweet.inReplyToUser, tweet.lang])

        # Creating a dataframe from the tweets list above
        self.tweets_df = pd.DataFrame(self.tweets_list,
                                      columns=['datetime', 'tweet_id', 'text', 'username', "quoted_tweet",
                                               "retweeted_tweet",
                                               "in_reply_to_user", "tweet_lang"])
        self.tweets_df.datetime = pd.to_datetime(self.tweets_df.datetime)
        return self.tweets_df

    def get_tweets_df(self):
        return self.tweets_df
