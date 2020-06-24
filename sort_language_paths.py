with open("best_returned_path.txt","r",encoding="latin-1") as in_file:
	language_pairs = [l.replace('(','').replace(')','').split() for l in in_file.readlines()]
sorted_language_pairs = sorted(language_pairs,key=lambda x: float(x[5])-float(x[2]) if x[5] != 'None' else -1,reverse=True)


with open("sorted_best_returned_path.txt","w+",encoding="latin-1") as out_file:
	for pair in sorted_language_pairs:
		out_file.write(" ".join(pair)+"\n")
