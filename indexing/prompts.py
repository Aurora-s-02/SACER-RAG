SUMMARIZE_DETAILS = """You are a helpful assistant that summarizes the details of a novel. You will be given a part of a novel. You need to summarize given content. The summary should include the main characters, the main plot and some other details. You need to return the summary in a concise manner without any additional fictive information. The length of the summary should be about 1000 tokens.
Here is the content:
Content: {content}
Now, please summarize the content.
Summary: """

SUMMARIZE_SUMMARY = """You are a helpful assistant that further summarizes the summaries of a novel. You will be given a series of summaries of parts of a novel. You need to summarize the summaries in a concise manner. The length of the summary should be about 1000 tokens.
Here is the summaries:
Summary: {summary}
Now, please summarize the summary based on the question.
Summary: """
