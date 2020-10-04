# Text Analysis


## Intro

* Used in many fields
  * Document classification
    * Directing messages to right place/dept
  * Search engines
    * Retrieve most suitable information
    
* When applied to the financial market:
  * Measuring risk and uncertainty
  * Assessing legal risk
  * Predicting asset prices
    * through processing specific pieces of text
  * Defining alternative industry classifications
  
> Text analysis and numerical data analytics are not fundamentally different. Many methods are applied similarly.
... _However, textual data is very different from numerical data._
  * Unstructured
  * Misspellings
  * Writing styles (abbreviations, punctuations)
  * Synonymns
  * Specific terminology
  * Context
  
## Processing text into vectors
* Collecting text
  * Web-scraping
  * Databases
  * APIs
  
### Reducing the difficulty of the task
> Reduce size of vector to a workable format
* Document standardisation
  * XML and HTML standardisation can help with extracting the text that we want
* Tokenisation
  * Break the document into terms
  * Usually using general tokenisation approaches but thees can also be customised
* Stemming examples
  * Conversions of each token to a standard form
    * e.g. reduction, reductions, reduce, reduced -> reduc
* Dimention reduction
  * Remove stop words _'words that do not have statistical power'_(e.g and, the, it, they)
  * remove very frequent words (does not differentiate documents) and very rare (typos)
* Create local dictionary
  * Focus on the words that are collected in the analysed subset of documents
  
_After doing all this processing we can present the document as a 'bag of words'_
  * Ignore grammar, word order, sentence structure, ...
  * Each word treated as a potentially important keywords
  
  
## Normalising textual data
* Term frequency
  * In the vector that we create from the bacg of words and the occurences of the documents we need to take into account the length of each document
    * The raw term frequencies are normalised by the length of the test. Represents the importance of the word in the text better.
  * A term that appears in all documents might still be important. If it is a generally rare word it will be helpful for retrieval purposes
  
* Inverse Document Frequency (IDF)
IDF(t) = log (Number of documents in collection / Number of documents containing term t)

* For a document we can take the normalised TF and multiply it by the IDF

