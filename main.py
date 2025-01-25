import os
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from search_tools import search_arxiv, search_pubmed

model = OpenAIModel(
    'anthropic/claude-3.5-sonnet',
    base_url='https://openrouter.ai/api/v1',
    api_key=os.getenv('OPENROUTER_API_KEY'),
)

agent = Agent(
    model,
    system_prompt='''You are a helpful research assistant. You can search for academic papers on arXiv and PubMed.
    Be concise in your responses and always include relevant paper titles and authors when answering research questions.''',
    tools=[search_arxiv, search_pubmed]
)

summary_agent = Agent(
    model,
    system_prompt='''Act as an expert summarizer with advanced NLP capabilities. Follow these steps meticulously:

    Preprocess the Input:
    Remove ads, duplicates, and irrelevant sections.
    Segment [CONTENT] into logical chunks (e.g., paragraphs, sections).
    Extract Core Information:
    Identify key entities, themes, and sentiment using NLP.
    Rank sentences by salience (position, semantic relevance, keyword frequency).
    Choose Summarization Strategy:
    If the content is technical/factual (e.g., research, news), use extractive summarization (direct quotes of key sentences).
    If the content is narrative/creative (e.g., stories, opinions), use abstractive summarization (paraphrase concepts).
    Optimize for Context:
    Adapt to the [DOMAIN] (e.g., medical, legal, tech) for terminology accuracy.
    Resolve ambiguities (e.g., link pronouns like 'it' or 'they' to their entities).
    Structure the Summary:
    Length: [DESIRED LENGTH] (e.g., 10% of original, 3 sentences, or 300 words).
    Format: Start with a 1-sentence TL;DR, then expand key points hierarchically.
    Use connectors like "However," "Additionally," or "In conclusion" for flow.
    Validate & Refine:
    Cross-check critical claims against verified sources (if available).
    Remove redundancies and simplify jargon.
    Preserve the original tone (e.g., formal, casual) and intent.
    Ethical Guardrails:
    Audit for bias (gender, race, etc.) and neutralize if detected.
    Never introduce new claims unsupported by the original content.
    
    Proceed step-by-step and explain your reasoning before delivering the final summary."*.',
    '''    
    )


content = '''

		
	
View in browser
WILL TRUMP SIDE WITH THE HARDLINERS ON RUSSIA?
And unspeakable lows in presidential pardons
Seymour Hersh
Jan 23
		
∙
		
Paid
	
 
		
		
		
	
READ IN APP
 
	
	
Donald Trump is sworn in as the 47th US President in the US Capitol Rotunda in Washington, DC, on January 20, 2025. / Photo by SAUL LOEB/POOL/AFP via Getty Images.

I thought we had hit bottom over the past two years when it was clear to me, and obvious to other journalists, that President Joe Biden was beginning to falter and that it was being covered up by many of his staff. I was told, directly and indirectly, about incidents known to those who served with him in the Senate that made it clear that the president’s memory was going. A journalist who spent time two years ago on Air Force Once with the president and his immediate family told of witnessing Biden unable to finish a sentence. There were repeated accounts by old pals about calls from the president that could not be returned because the President’s calls were being monitored by his staff. The process was a shameful conspiracy of silence that ended with Biden’s evident confusion during his disastrous debate with Trump on June 27. His performance made his much delayed decision to not run for a second term inevitable. It came too late to have a candidate selected by a primary process or an open convention.

All of this happened despite the findings three months earlier by special Justice Department counsel Robert Hur, a former Supreme Court clerk, dealing with evidence that the president had “willfully retained and disclosed classified materials after his vice presidency when he was a private citizen.” Hur’s report revealed that Biden kept classified documents, some of them top secret, scattered all over his various offices. But he concluded that it would be hard to prosecute Biden for these violations because, among other issues, he came across in testimony as a “sympathetic, well-meaning elderly man with a poor memory. . . . It would be difficult to convince a jury that they should convict him—by then a former president well into his eighties—of a serious felony that requires a mental state of willfulness.” Hur gave no interviews at the time his report was released and has consistently refused to discuss the issue since.

This month there were Biden’s last-minute pardons of many members of his immediate family, including his two brothers, along with the members of his government and the military who were said by Donald Trump to be on his list to be investigated and prosecuted when he came to office. The unprecedented pardon of family members and political supporters came twenty minutes before the president left office for the final time on Monday, inauguration day.

And then Trump, newly sworn in, sank lower by doing what he said he would do hours after his inauguration Monday—he pardoned all the rioters who on January 6, 2021, attacked police and broke in and ransacked the Congress in what clearly was an unprecedented attempt to prevent the Senate from certifying Biden’s election.

Among those pardoned were some found guilty of assaulting Capitol police officers who did what they could do in a futile but heroic effort to protect the Congress and the legislators inside from the thousands of demonstrators who, believing they were doing what Trump wanted them to do, broke through windows to gain entrance to the Halls of Congress.

Trump’s pardon came a few days after he had accomplished what Biden and his foreign policy aides had failed to do: convince Israeli Prime Minister Benjamin Netanyahu to agree to a long-sought ceasefire. The beginning of a process to recover Israeli hostages was not the culmination of months of negotiation by Biden’s senior foreign policy aides, as they later suggested to newspaper reporters, but Netanyahu’s understanding that there was a new sheriff in town.

It’s not clear whether Netanyahu will agree to the second and third phases of the agreement, which call for all the 10/7 hostages to be released and an end to the Israeli military occupation of Gaza, but for now the future is in Trump’s hands. He has delivered what Biden could not get. The bombing of Gaza has stopped—though the increasingly embattled West Bank is still under fire—and food trucks are pouring into Gaza.

At this early point, there seem to be many competing interests among Trump’s close foreign policy advisers on the second major issue the new Administration will confront: how and when to end the ongoing war between Russia and Ukraine.

During his campaign, Trump repeatedly vowed to end the Ukraine War even before taking office. It’s easy to mock those statements now, but in my reporting I have been told by someone with firsthand information that intense talks between Ukraine and Russia are ongoing and have moved “close to a settlement.”

Right now one of the main issues involves what I was told is “jockeying for territory.” Ukrainian President Volodymyr “Zelensky has to save face,” a knowledgeable American told me. “He never wants to kneel to the Russians.”

The war has been brutal, with enormous casualties to front-line soldiers on both sides. The issues boil down to how much territory Russia will retain in the provinces where it continues to make small gains in trench warfare against the undermanned and under-equipped Ukrainian forces. “Putin is the bully In the schoolyard,” the American said, “and we gotta say to the Russians: ‘Let’s talk about what you’re going to get.’” In some places in Ukraine, he said, a negotiating issue comes down to whether a specific smelting plant would be Russian or Ukrainian.

It was his understanding that Trump initially was on board with the negotiations, and his view was that no settlement would work unless Putin was left with “a way to make money” in return for agreeing to end the war. Trump, the American said, “knows nothing about international history,” but he does understand that Putin, whose economy is staggering under heavy sanctions and an inflation rate of 8.5 percent, is in urgent need of finding more markets for his nation’s vast gas and oil reserves.

The advanced state of the negotiations was being monitored, I was told, by senior US generals and Trump campaign aides, all to be fixtures in Trump’s government. Amid what seemed to be a path to the end of the war, came a little-noted announcement on January 8 by retired Army Lieutenant General Keith Kellogg, a conservative who served in Trump’s first administration and now is Trump’s special envoy for the current peace talks between Ukraine and Russia. Kellogg, publicly contradicting the president-elect, told Fox News that the war would not end with Trump’s arrival in office but could be resolved within one hundred days of his inauguration. “This is a war that needs to end,” Kellogg said, “and I think he can do it in the near term.” (Trump had made another timeline statement for ending the Ukraine war the day before in a chaotic press conference at Mar-a-Lago, but his words were lost amid his claim that he could end the Ukraine War in six months and would not have a summit meeting with Putin until after he took office.)

I was told by a person with access to current thinking in the Trump camp that the president-elect had come to understand that he had spoken too soon about the possibility of an agreement over Ukraine with Putin. Among the reasons for delaying serious talks was the belief that NATO countries will be persuaded by Trump to increase their annual payments to NATO, in some cases more than doubling their annual 2 percent contribution of gross annual income. I was further told that Trump wants the larger European countries to raise that number to 5 percent. If that came to pass, NATO funding would be increased by billions of dollars and a better financed NATO “would be seen as a threat to Putin.” The underlying point is that some of Trump’s advisers believe Putin “wants more of Ukraine that he will get.” And without more NATO support, it is believed that “Putin will not learn the folly of attacking the West.”

The hardline view sees Putin as an inevitable aggressor who has been successful: in Russia’s invasion of Georgia in 2008; in the seizure of Crimea in 2014; in the 2022 war in Ukraine; and in its continuing support of Iran, whose continuing enrichment of uranium—all under the camera monitoring of the International Atomic Energy Agency in Vienna. All this is viewed with alarm by many in the Trump administration.

Another issue is Russian support for BRICs—the alternative international trade and energy group that includes Brazil, Russia, Iran, China, and South Africa that is viewed as a potential economic threat to the West’s G7 community. The ultimate fear of some in the West, and in the White House, I was told, is that “Russia and China will try to infuse BRICs with a military component” along with creating an international alternative to the dollar.

In the American conservative view, delaying a settlement between Russia and Ukraine could offer the West a chance to minimize the growth of BRICs. The new Trump administration should not rush to an agreement with Russia and end the Ukraine war in its murderous stalemate and instead send Putin and his allies in China and elsewhere a message: “The more you want in the Ukraine war the more you will lose.”

Washington, and America, is now in the hands of the usually marginalized hard liners. Where will Donald Trump, who so much wants to be loved, end up?

Invite your friends and earn rewards
If you enjoy Seymour Hersh, share it with your friends and earn rewards when they subscribe.

Invite Friends
 
Like
		
Comment
		
Restack
	
 

© 2025 Seymour Hersh
Unsubscribe

Get the appStart writing
	
'''

def main():
    #theme = input("Enter your research topic: ")
    #print(f"\nSearching for papers about: '{theme}'...\n")
    #result = agent.run_sync(f'Find recent papers about {theme} and their applications. Use the tools pubmed only for medical papers. Please state the tool you used in your response.')
    #print(result.data)

    print("\nSummarizing the content...\n")
    summary_result = summary_agent.run_sync(f'Summarize the following content: {content}')
    print(summary_result.data)

if __name__ == "__main__":
    main()
