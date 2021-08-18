with open('users.csv', 'r') as infile:
	contents = infile.read()
	contents.replace(u'\ufeff', '')
	
with open('users2.csv', 'w') as outfile:
	outfile.write(contents)