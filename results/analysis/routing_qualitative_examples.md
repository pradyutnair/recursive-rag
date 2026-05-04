# Routing Qualitative Examples

## Routing helped

### 1. dataset / one_hop / easy
Q: Which country Aleksander Koniecpolski (1620–1659)'s father is from?
Gold: Polish-Lithuanian Commonwealth
Adaptive: Polish-Lithuanian Commonwealth (EM=1.0, tokens=2054)
Force-hard: Polish (EM=0.0, tokens=5160)
Token delta adaptive-hard: 3106

### 2. dataset / yes_no / easy
Q: Are the movies Wizards Of The Lost Kingdom and Final Exam (1981 Film), from the same country?
Gold: no
Adaptive: no (EM=1.0, tokens=1814)
Force-hard: yes (EM=0.0, tokens=6030)
Token delta adaptive-hard: 4216

### 3. dataset / yes_no / easy
Q: Are Wiqu (Lima) and Qillqa (Melgar) both located in the same country?
Gold: yes
Adaptive: yes (EM=1.0, tokens=1944)
Force-hard: no (EM=0.0, tokens=7833)
Token delta adaptive-hard: 5889

### 4. dataset / one_hop / easy
Q: Who was born earlier, Wang Shusen or Rona Randall?
Gold: Rona Randall
Adaptive: Rona Randall (EM=1.0, tokens=1803)
Force-hard: Wang Shusen and Rona Randall have the same birth date of 16 June 1911 (EM=0.0, tokens=18089)
Token delta adaptive-hard: 16286

### 5. dataset / one_hop / easy
Q: Which film was released earlier, Dreamville Presents: Revenge or One More Time With Feeling?
Gold: One More Time With Feeling
Adaptive: One More Time with Feeling (EM=1.0, tokens=1781)
Force-hard: Dreamville Presents: Revenge (EM=0.0, tokens=12222)
Token delta adaptive-hard: 10441

## Routing hurt

### 1. dataset / one_hop / easy
Q: What county is Erik Hort's birthplace a part of?
Gold: Rockland County
Adaptive: Montebello (EM=0.0, tokens=1878)
Force-hard: Rockland County (EM=1.0, tokens=4749)
Token delta adaptive-hard: 2871

### 2. dataset / temporal / easy
Q: In what year was the group who performed Attics To Eden formed?
Gold: 2005
Adaptive: 1987 (EM=0.0, tokens=1833)
Force-hard: 2005 (EM=1.0, tokens=4656)
Token delta adaptive-hard: 2823

### 3. dataset / one_hop / easy
Q: What year did the city were Hans-Joachim Merker was born end?
Gold: 1738
Adaptive: 1929 (EM=0.0, tokens=1863)
Force-hard: 1738 (EM=1.0, tokens=4767)
Token delta adaptive-hard: 2904

### 4. dataset / one_hop / easy
Q: What team was Anna Benson's husband on?
Gold: Pittsburgh Pirates
Adaptive: New York Mets (EM=0.0, tokens=1771)
Force-hard: Pittsburgh Pirates (EM=1.0, tokens=4567)
Token delta adaptive-hard: 2796

### 5. dataset / temporal / easy
Q: What year did the group that performed From Them, Through Us, to You form?
Gold: 2005
Adaptive: 2007 (EM=0.0, tokens=1871)
Force-hard: 2005 (EM=1.0, tokens=4805)
Token delta adaptive-hard: 2934
