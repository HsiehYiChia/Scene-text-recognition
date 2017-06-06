/*
* SpellingCorrector.h
*
* Version: 1.5
* Author: Felipe Farinon <felipe.farinon@gmail.com>
* URL: https://scarvenger.wordpress.com/
*
* Changelog:
* 1.5: corrected transpositions in edits method and word load in load method.
*/

#ifndef _SPELLINGCORRECTOR_H_
#define _SPELLINGCORRECTOR_H_

#include <vector>
#include <map>

class SpellingCorrector
{
private:
	typedef std::vector<std::string> Vector;
	typedef std::map<std::string, int> Dictionary;

	Dictionary dictionary;

	void edits(const std::string& word, Vector& edited_str);
	void known(Vector& edited_str, Dictionary& candidates);

public:
	void load(const std::string& filename);
	std::string correct(const std::string& word);
};

#endif
