sed 's/[^a-zA-ZÀÈÌÒÙàèìòùÁÉÍÓÚáéíóúÂÊÎÔÛâêîôûÄËÏÖÜäëïöüÇçŒœ]\+/\n/g' | grep -v '^$' > $$words
tail -n +2 $$words > $$nextwords
paste $$words $$nextwords |
sort | uniq -c
# remove the temporary files
rm $$words $$nextwords
