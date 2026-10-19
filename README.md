# Evolutionärer Algorithmus
Python-Anwendung, welche die Meta-Heuristik eines evolutionären Algorithmus implementiert zur Ermittlung der flachsten Parabel (y=ax2+bx+c), durch Kombination der Werte für a, b und c.

## Terminologie der Meta-Heuristik nach diesem Anwendungsfall
* Gen: Es gibt drei Gene: a, b und c
* Chromosom/Individuum: Ist das Triplett aus den drei Genen. In diesem Kontext ist Chromosom und Individuum gleich. Normalerweise werden sie aber unterschieden.
* Phänotyp: Ist die geplottete Funktion.
* Population: Ist die Gesamtheit aller Individuen.

## Abbruchbedingungen
* Erreichen eines Zielfitnesswertes
* Erreichen einer fest definierten Anzahl an Generationen.
* Stagnation -> Fitnesswert verbessert sich über bestimmte Anzahl an Generationen nicht.

## Pseudocode
* Aktuelle Population = Ausgangspopulation
* Bewerte Indivividuen in Ausgangspopulation
* Wiederhole (Erzeuge neue Generation)
    * Selektion zur Reproduktion aus aktueller Population
    * Replikation
    * Variation
    * Bewerte Individuen der neuen Population
    * Selektion für Überleben und Bilden neuer aktueller Population
Bis Abbruchkriterium erreicht

## Fitnessfunktion
Der Fitnesswert ist das negierte Ergebnis der Krümmungsfunktion. Die Krümmungsfunktion \(κ(x)\) gibt die Stärke der Krümmung an.
\( κ(x)=\left| \frac{f''(x)}{(1+(f'(x))^2)^(3/2)} \right| \)


## Ausgaben
Es wird die Funktion im Koordinatensystem geplotet ...
![Funktion im Koordinatensystem](/docs/Screenshots/Fig1.png "Funktion")

... die Entwicklung der Gene über die Generationen hinweg ...
![Gen-Entwicklung](/docs/Screenshots/Fig2.png "Gene")

... und Verbesserung des Fitnesswertes sowie sein Min-Max-Bereich über die Generationen hinweg.
![Fitness-Entwicklung](/docs/Screenshots/Fig3.png "Zeitreihe")

Des Weiteren werden in der Konsole der Grund für den Abbruch des Programms angegeben, sowie die Gene des besten Individuum als auch die Generation, in welchem dieses Individuum entstand.