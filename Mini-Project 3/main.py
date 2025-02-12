from SentenceReadingAgent import SentenceReadingAgent

def test():
    #This will test your SentenceReadingAgent
	#with nine initial test cases.

    test_agent = SentenceReadingAgent()

    sentence_1 = "Ada brought a short note to Irene."
    question_1 = "Who brought the note?"
    question_2 = "What did Ada bring?"
    question_3 = "Who did Ada bring the note to?"
    question_4 = "How long was the note?"

    sentence_2 = "David and Lucy walk one mile to go to school every day at 8:00AM when there is no snow."
    question_5 = "Who does Lucy go to school with?"
    question_6 = "Where do David and Lucy go?"
    question_7 = "How far do David and Lucy walk?"
    question_8 = "How do David and Lucy get to school?"
    question_9 = "At what time do David and Lucy walk to school?"

    print(test_agent.solve(sentence_1, question_1))  # "Ada"
    print(test_agent.solve(sentence_1, question_2))  # "note" or "a note"
    print(test_agent.solve(sentence_1, question_3))  # "Irene"
    print(test_agent.solve(sentence_1, question_4))  # "short"

    print(test_agent.solve(sentence_2, question_5))  # "David"
    print(test_agent.solve(sentence_2, question_6))  # "school"
    print(test_agent.solve(sentence_2, question_7))  # "mile" or "a mile"
    print(test_agent.solve(sentence_2, question_8))  # "walk"
    print(test_agent.solve(sentence_2, question_9))  # "8:00AM"

    sentence_test = 'Their children are in school'
    question_test = 'Where are their children'
    print(test_agent.solve(sentence_test, question_test))

    sentence_test = 'Watch your step'
    question_test = 'What should you watch' 
    print(test_agent.solve(sentence_test, question_test))

    sentence_test = 'Lucy will write a book'
    question_test = 'What will Lucy do'
    print(test_agent.solve(sentence_test, question_test))

    sentence_test = 'The blue bird will sing in the morning'
    question_test = 'What will sing in the morning'
    print(test_agent.solve(sentence_test, question_test))

    sentence_test = 'Give us all your money'
    question_test = 'Who should you give your money to?'    
    print(test_agent.solve(sentence_test, question_test))

    sentence_test = 'It is a small world after all'
    question_test = 'How big is the world?'   
    print(test_agent.solve(sentence_test, question_test))

    sentence_test = 'There are a thousand children in this town'
    question_test = 'How many children are in this town'
    print(test_agent.solve(sentence_test, question_test))

    sentence_test = 'Serena saw a home last night with her friend'
    question_test = 'Who was with Serena'
    print(test_agent.solve(sentence_test, question_test))

if __name__ == "__main__":
    test()