from transformers import pipeline
def short_textBart(document):
    summerizer = pipeline('summarization')
    my_summary = summerizer(document, max_length=500, min_length=200, do_sample=False)
    print(my_summary[0]['summary_text'])



article = '''

Chapter I


IN WHICH PHILEAS FOGG AND PASSEPARTOUT ACCEPT EACH OTHER, 
THE ONE AS
MASTER, THE OTHER AS MAN
Mr. Phileas Fogg lived, in 1872, at No. 7, Saville Row, Burlington
Gardens, the house in which Sheridan died in 1814.  He was one of the
most noticeable members of the Reform Club, though he seemed always to
avoid attracting attention; an enigmatical personage, about whom little
was known, except that he was a polished man of the world.  People said
that he resembled Byron—at least that his head was Byronic; but he was
a bearded, tranquil Byron, who might live on a thousand years without
growing old.
Certainly an Englishman, it was more doubtful whether Phileas Fogg was
a Londoner.  He was never seen on 'Change, nor at the Bank, nor in the
counting-rooms of the "City"; no ships ever came into London docks of
which he was the owner; he had no public employment; he had never been
entered at any of the Inns of Court, either at the Temple, or Lincoln's
Inn, or Gray's Inn; nor had his voice ever resounded in the Court of
Chancery, or in the Exchequer, or the Queen's Bench, or the
Ecclesiastical Courts.  He certainly was not a manufacturer; nor was he
a merchant or a gentleman farmer.  His name was strange to the
scientific and learned societies, and he never was known to take part
in the sage deliberations of the Royal Institution or the London
Institution, the Artisan's Association, or the Institution of Arts and
Sciences.  He belonged, in fact, to none of the numerous societies
which swarm in the English capital, from the Harmonic to that of the
Entomologists, founded mainly for the purpose of abolishing pernicious
insects.
Phileas Fogg was a member of the Reform, and that was all.
The way in which he got admission to this exclusive club was simple
enough.
He was recommended by the Barings, with whom he had an open credit.
His cheques were regularly paid at sight from his account current,
which was always flush.
Was Phileas Fogg rich?  Undoubtedly.  But those who knew him best could
not imagine how he had made his fortune, and Mr. Fogg was the last
person to whom to apply for the information.  He was not lavish, nor,
on the contrary, avaricious; for, whenever he knew that money was
needed for a noble, useful, or benevolent purpose, he supplied it
quietly and sometimes anonymously.  He was, in short, the least
communicative of men.  He talked very little, and seemed all the more
mysterious for his taciturn manner.  His daily habits were quite open
to observation; but whatever he did was so exactly the same thing that
he had always done before, that the wits of the curious were fairly
puzzled.

Had he travelled?  It was likely, for no one seemed to know the world
more familiarly; there was no spot so secluded that he did not appear
to have an intimate acquaintance with it.  He often corrected, with a
few clear words, the thousand conjectures advanced by members of the
club as to lost and unheard-of travellers, pointing out the true
probabilities, and seeming as if gifted with a sort of second sight, so
often did events justify his predictions.  He must have travelled
everywhere, at least in the spirit.

It was at least certain that Phileas Fogg had not absented himself from
London for many years.  Those who were honoured by a better
acquaintance with him than the rest, declared that nobody could pretend
to have ever seen him anywhere else.  His sole pastimes were reading
the papers and playing whist.  He often won at this game, which, as a
silent one, harmonised with his nature; but his winnings never went
into his purse, being reserved as a fund for his charities.  Mr. Fogg
played, not to win, but for the sake of playing.  The game was in his
eyes a contest, a struggle with a difficulty, yet a motionless,
unwearying struggle, congenial to his tastes.







'''

short_textBart(article)