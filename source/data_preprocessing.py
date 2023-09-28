import re
import texthero
import nltk.stem.porter
import text_preprocessing


class DataPreprocessing:
    
    def __init__(self):
        pass

    def clean_text_blocks(self, df, attribute, level=None):
        if level == "minimal":
            df[attribute] = df[attribute].apply(lambda x: re.sub("[\n]", "", x))
            df[attribute] = df[attribute].apply(lambda x: text_preprocessing.preprocess_text(x, [
                text_preprocessing.to_lower, text_preprocessing.remove_number]))
        if level == "heavy":     
            df[attribute] = texthero.clean(df[attribute])
        return df

    def filter_text_blocks(self, df, attribute, keep_only_format=None, keep_only_size=None, keep_only_keywords=None):
        if keep_only_format:
            df = df[df["Content Type"] == keep_only_format].copy()
        if keep_only_size:  
            min_length, max_length = keep_only_size
            df["length"] = df[attribute].str.len()
            df = df[((df["length"] >= min_length) & (df["length"] <= max_length))].copy()
            df.drop("length", axis=1, inplace=True)
        if keep_only_keywords:    
            stemmer = nltk.stem.porter.PorterStemmer()
            stemmed_words = [stemmer.stem(kw) for kw in keep_only_keywords]
            pattern = "|".join(stemmed_words)
            df = df[df[attribute].str.contains(pattern, case=False)].copy()
        return df