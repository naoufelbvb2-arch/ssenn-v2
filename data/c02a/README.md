# Poetry corpus

`blake.json` contains 45 poem bodies from William Blake's *Songs of Innocence and of Experience*, in source order. The source is the Project Gutenberg transcription distributed in the NLTK data collection. The local snapshot makes the experiment runnable offline. [Pinned NLTK archive](https://github.com/nltk/nltk_data/blob/550b6625bcef1f2abff2ff770a5a0d272c9c6b2a/packages/corpora/gutenberg.zip)

The archive member `gutenberg/blake-poems.txt` has SHA-256 `840897a66e23a47dd1fd56b864f421b2be0af074dc081244f7995bf26939eee9`. Starting at the standalone heading SONGS OF INNOCENCE, use each uppercase poem heading as a boundary and the two collection headings as sections. Stop before APPENDIX. Join whitespace-separated body text with single spaces. Section and title identify the document and are excluded from model inputs. Case folding occurs during tokenization. The configured split is over complete poems and uses 33 for training, six for validation and six for test.
