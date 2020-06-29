# multilingual-routing

Project exploring translation between a pair of languages, **A->C** via a third (Or more) pivot language(s), **B_0,B_1...**.

Research Qs: 
- What sort of perfomance boosts can we get from routing? (Can we outweigh the cost of compounded error?)
- Can we predict how a given input sentence should be routed?


# TODO:
- Explore regression-based approaches, predict scores independently
- Explore alternative pivots. Here are the best candidates, ranked:

<pre><code>
Original: 'ase-de', 27.2 Other: 'en', 73.4
Original: 'en-ig', 3.8 Other: 'sv', 45.6
Original: 'fj-fr', 24.0 Other: 'en', 65.1
Original: 'ase-sv', 39.7 Other: 'en', 79.8
Original: 'sv-el', 20.8 Other: 'en', 60.45
Original: 'lg-en', 5.4 Other: 'sv', 44.5
Original: 'en-ee', 6.0 Other: 'sv', 44.9
Original: 'el-sv', 23.6 Other: 'fr', 61.55
Original: 'ase-fr', 37.8 Other: 'en', 75.0
Original: 'sv-nl', 24.3 Other: 'en', 60.8
Original: 'sv-et', 23.5 Other: 'en', 59.25
Original: 'sv-fj', 27.8 Other: 'en', 63.5
Original: 'nl-sv', 25.0 Other: 'en', 60.5
Original: 'tr-sv', 26.3 Other: 'en', 61.8
Original: 'en-lg', 5.7 Other: 'sv', 41.15
Original: 'lv-sv', 22.0 Other: 'en', 56.7
Original: 'de-et', 20.2 Other: 'en', 54.7
Original: 'de-fj', 24.6 Other: 'en', 58.95
Original: 'uk-sv', 27.8 Other: 'en', 62.099999999999994
  </code></pre>
