import time
import calendar

from guesser import Guesser
from collections import defaultdict
from nltk.tokenize.treebank import TreebankWordTokenizer

kPRESIDENT_DATA = {"train": [
  {"start": 1789, "start_date":[4, 30], "stop": 1797, "end_date":[3, 4], "name": "George Washington"},
  {"start": 1797, "start_date":[3, 4], "stop": 1801, "end_date":[3, 4], "name": "John Adams"},
  {"start": 1801, "start_date":[3, 4], "stop": 1809, "end_date":[3, 4], "name": "Thomas Jefferson"},
  {"start": 1809, "start_date":[3, 4], "stop": 1817, "end_date":[3, 4], "name": "James Madison"},
  {"start": 1817, "start_date":[3, 4], "stop": 1825, "end_date":[3, 4], "name": "James Monroe"},
  {"start": 1825, "start_date":[3, 4], "stop": 1829, "end_date":[3, 4], "name": "John Quincy Adams"},
  {"start": 1829, "start_date":[3, 4], "stop": 1837, "end_date":[3, 4], "name": "Andrew Jackson"},
  {"start": 1837, "start_date":[3, 4], "stop": 1841, "end_date":[3, 4], "name": "Martin Van Buren"},
  {"start": 1841, "start_date":[3, 4], "stop": 1841, "end_date":[4, 6], "name": "William Henry Harrison"},
  {"start": 1841, "start_date":[4, 6], "stop": 1845, "end_date":[3, 4], "name": "John Tyler"},
  {"start": 1845, "start_date":[3, 4], "stop": 1849, "end_date":[3, 4], "name": "James K. Polk"},
  {"start": 1849, "start_date":[3, 4], "stop": 1850, "end_date":[7, 10], "name": "Zachary Taylor"},
  {"start": 1850, "start_date":[7, 10], "stop": 1853, "end_date":[3, 4], "name": "Millard Fillmore"},
  {"start": 1853, "start_date":[3, 4], "stop": 1857, "end_date":[3, 4], "name": "Franklin Pierce"},
  {"start": 1857, "start_date":[3, 4], "stop": 1861, "end_date":[3, 4], "name": "James Buchanan"},
  {"start": 1861, "start_date":[3, 4], "stop": 1865, "end_date":[4, 15], "name": "Abraham Lincoln"},
  {"start": 1865, "start_date":[4, 15], "stop": 1869, "end_date":[3, 4], "name": "Andrew Johnson"},
  {"start": 1869, "start_date":[3, 4], "stop": 1877, "end_date":[3, 5], "name": "Ulysses S. Grant"},
  {"start": 1877, "start_date":[3, 5], "stop": 1881, "end_date":[3, 4], "name": "Rutherford Birchard Hayes"},
  {"start": 1881, "start_date":[3, 4], "stop": 1881, "end_date":[9, 20], "name": "James A. Garfield"},
  {"start": 1881, "start_date":[9, 20], "stop": 1885, "end_date":[3, 4], "name": "Chester A. Arthur"},
  {"start": 1885, "start_date":[3, 4], "stop": 1889, "end_date":[3, 4], "name": "Grover Cleveland"},
  {"start": 1889, "start_date":[3, 4], "stop": 1893, "end_date":[3, 4], "name": "Benjamin Harrison"},
  {"start": 1893, "start_date":[3, 4], "stop": 1897, "end_date":[3, 4], "name": "Grover Cleveland"},
  {"start": 1897, "start_date":[3, 4], "stop": 1901, "end_date":[9, 14], "name": "William McKinley"},
  {"start": 1901, "start_date":[9, 14], "stop": 1905, "end_date":[3, 4], "name": "Theodore Roosevelt"},
  {"start": 1905, "start_date":[3, 4], "stop": 1909, "end_date":[3, 4], "name": "Theodore Roosevelt"},
  {"start": 1909, "start_date":[3, 4], "stop": 1913, "end_date":[3, 4], "name": "William H. Taft"},
  {"start": 1913, "start_date":[3, 4], "stop": 1921, "end_date":[3, 4], "name": "Woodrow Wilson"},
  {"start": 1921, "start_date":[3, 4], "stop": 1923, "end_date":[8, 3], "name": "Warren G. Harding"},
  {"start": 1923, "start_date":[8, 3], "stop": 1929, "end_date":[3, 4], "name": "Calvin Coolidge"},
  {"start": 1929, "start_date":[3, 4], "stop": 1933, "end_date":[3, 4], "name": "Herbert Hoover"},
  {"start": 1933, "start_date":[3, 4], "stop": 1945, "end_date":[4, 12], "name": "Franklin D. Roosevelt"},
  {"start": 1945, "start_date":[4, 12], "stop": 1953, "end_date":[1, 20], "name": "Harry S. Truman"},
  {"start": 1953, "start_date":[1, 20], "stop": 1961, "end_date":[1, 20], "name": "Dwight D. Eisenhower"},
  {"start": 1961, "start_date":[1, 20], "stop": 1963, "end_date":[11, 22], "name": "John F. Kennedy"},
  {"start": 1963, "start_date":[11, 22], "stop": 1969, "end_date":[1, 20], "name": "Lyndon B. Johnson"},
  {"start": 1969, "start_date":[1, 20], "stop": 1974, "end_date":[8, 9], "name": "Richard M. Nixon"},
  {"start": 1974, "start_date":[8, 9], "stop": 1977, "end_date":[1, 20], "name": "Gerald R. Ford"},
  {"start": 1977, "start_date":[1, 20], "stop": 1981, "end_date":[1, 20], "name": "Jimmy Carter"},
  {"start": 1981, "start_date":[1, 20], "stop": 1989, "end_date":[1, 20], "name": "Ronald Reagan"},
  {"start": 1989, "start_date":[1, 20], "stop": 1993, "end_date":[1, 20], "name": "George Bush"},
  {"start": 1993, "start_date":[1, 20], "stop": 2001, "end_date":[1, 20], "name": "Bill Clinton"},
  {"start": 2001, "start_date":[1, 20], "stop": 2009, "end_date":[1, 20], "name": "George W. Bush"},
  {"start": 2009, "start_date":[1, 20], "stop": 2017, "end_date":[1, 20], "name": "Barack Obama"},
  {"start": 2017, "start_date":[1, 20], "stop": 2021, "end_date":[1, 20], "name": "Donald J. Trump"},
  {"start": 2021, "start_date":[1, 20], "stop": 2025, "end_date":[1, 20], "name": "Joseph R. Biden"}],
  "dev": [{"text": "Who was president on Wed Jan 25 06:20:00 2023?", "page": "Joseph R. Biden", "qanta_id":201},
          {"text": "Who was president on Sat May 23 02:00:00 1982?", "page": "Ronald Reagan", "qanta_id":202},
          {"text": "Who was president on Wed Mar 01 04:23:40 2023?", "page": 'Joseph R. Biden', "qanta_id":203},
          {"text": "Who was president on Tue Jan 20 13:00:00 2009?", "page": 'Barack Obama', "qanta_id":204},
          {"text": "Who was president on Fri Nov 22 16:00:00 1963?", "page": 'Lyndon B. Johnson', "qanta_id":205},
          {"text": "Who was president on Tue Apr 12 20:00:00 1949?", "page": 'Harry S. Truman', "qanta_id":206},
          {"text": "Who was president on Sat Mar 04 21:00:00 1933?", "page": 'Franklin D. Roosevelt', "qanta_id":207},
          {"text": "Who was president on Sat Apr 15 15:00:00 1865?", "page": 'Andrew Johnson', "qanta_id":208},
          {"text": "Who was president on Thu Apr 30 17:00:00 1789?", "page": 'George Washington', "qanta_id":209}]
}

class PresidentGuesser(Guesser):
    def train(self, training_data):
        self._lookup = defaultdict(dict)
            
    def __call__(self, question, n_guesses=1):
        # Update this code so that we can have a different president than Joe
        # Biden
        t = TreebankWordTokenizer()
        toks = t.tokenize(question)
        year = int(toks[8])
        month = list(calendar.month_abbr).index(toks[5])
        day = int(toks[6])
        time = [int(x) for x in str(toks[7]).split(':')]
        candidates = []
        for p in kPRESIDENT_DATA["train"]:
            if p["start"] == p["stop"] and p["start"] == year:
                if p["start_date"][0] < month or (p["start_date"][0] == month and (p["start_date"][1] < day or (p["start_date"][1] == day and time[0] >= 12))) and p["end_date"][0] > month or (p["end_date"][0] == month and (p["end_date"][1] > day or (p["end_date"][1] == day and time[0] < 12))):
                    candidates.append(p["name"])
            elif p["start"] == year and (p["start_date"][0] < month or (p["start_date"][0] == month and (p["start_date"][1] < day or (p["start_date"][1] == day and time[0] >= 12)))):
                candidates.append(p["name"])
            elif p["stop"] == year and (p["end_date"][0] > month or (p["end_date"][0] == month and (p["end_date"][1] > day or (p["end_date"][1] == day and time[0] < 12)))):
                candidates.append(p["name"])
            elif p["start"] < year and p["stop"] > year:
                candidates.append(p["name"])
        if len(candidates) == 0:
            return [{"guess": ""}]
        else:
            return [{"guess": x} for x in candidates]
        
if __name__ == "__main__":
    pg = PresidentGuesser()

    pg.train(kPRESIDENT_DATA["train"])
    
    for date in kPRESIDENT_DATA["dev"]:
        print(date, pg(date)["guess"])
        
