# text-summarizer

Pipeline: 
1. split text into sentences
2. split sentences into words
3. delete articles, particles etc. from sentences
4. count tf * idf for every word in sentence
5. sum tf * idf over sentence for each sentence
6. order by sum tf * idf desc
7. pick first n sentences < 300 symbols
8. reorder picked sentences with natural order
