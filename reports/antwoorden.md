# Tentamen ML2022-2023

De opdracht is om de audio van 10 cijfers, uitgesproken door zowel mannen als vrouwen, te classificeren. De dataset bevat timeseries met een wisselende lengte.

In [references/documentation.html](references/documentation.html) lees je o.a. dat elke timestep 13 features heeft.
Jouw junior collega heeft een neuraal netwerk gebouwd, maar het lukt hem niet om de accuracy boven de 67% te krijgen. Aangezien jij de cursus Machine Learning bijna succesvol hebt afgerond hoopt hij dat jij een paar betere ideeen hebt.

## Vraag 1

### 1a
In `dev/scripts` vind je de file `01_model_design.py`.
Het model in deze file heeft in de eerste hidden layer 100 units, in de tweede layer 10 units, dit heeft jouw collega ergens op stack overflow gevonden en hij had gelezen dat dit een goed model zou zijn.
De dropout staat op 0.5, hij heeft in een blog gelezen dat dit de beste settings voor dropout zou zijn.

- Wat vind je van de architectuur die hij heeft uitgekozen (een Neuraal netwerk met drie Linear layers)? Wat zijn sterke en zwakke kanten van een model als dit in het algemeen? En voor dit specifieke probleem?
#### <span style="background-color: #197d2b">Antwoord:</span>
Een Neural Netwerk met Linear Layers is een relatief simpel model (In_features -> size of each input, out_features –> size of each output en een bias). Doordat het een (relatief) simpel model is dat helpt het om overfitting te voorkomen. Vanwege de simpliciteit en snelheid is het een goed basismodel om mee te starten. Dit is ook direct het grote nadeel van dit model. Doordat het een (algemeen) simpel model is behaald het niet altijd de hoogt mogelijke nauwkeurigheid. Kijkend naar de data en de vraag zal er dus gekeken moeten worden naar een meer specifiek model om een hogere nauwkeurigheid te behalen.
Voor dit specifieke probleem, zijnde classificatie van audio, is een model zoals deze niet de beste keuze. Om een hogere nauwkeurigheid te behalen kan er gekeken worden naar convolutional neural networks (CNNs) of misschien zelfs beter: Recurrent Neural Networks (RNN). RNN zijn specifiek goed in sequentiële gegevens zoals tekst, audio en video. 


- Wat vind je van de keuzes die hij heeft gemaakt in de LinearConfig voor het aantal units ten opzichte van de data? En van de dropout?
#### <span style="background-color: #197d2b">Antwoord:</span>
De vraagstelling vanuit de collega is om de data te classificeren. De data bestaat uit getalen van nul tot negen (n=10) uitgesproken in het Arabic door mannelijke en vrouwelijke (cat. n=2) sprekers. Dit betekent dat er in totaal 20 classes zijn die geïdentificeerd dienen te worden. Kijkend naar het geschreven model zien we het volgende:
```
(Getalen overgenomen om het leesbaar te maken)
nn.Linear(config["13"], config["100"]),
nn.ReLU(),
nn.Linear(config["100"], config["10"]),
nn.Dropout(config["0.5"]),
nn.ReLU(),
nn.Linear(config["10"], config["20"]),
```
De stappen die gemaakt worden in dit model zijn nogal groot. Van 13 units naar 100 units is een vrij grote stapt. Vervolgens wordt er een rectified linear unit (ReLU) toegepast als activatie functie (f(x)=max(0,x). De volgende stap is ook erg groot namelijk van 100 units terug naar 10 units. Daarna wordt er een Dropout functie toegepast om overfitting te voorkomen. Ook deze staat erg hoog afgesteld namelijk op 0.5 (i.e. de helft van de neurons wordt uitgeschakeld). Volgens de literatuur staat de dropout gebruikelijk tussen de 0.2 en 0.5. Na de dropout wordt er opnieuw een ReLU toegepast. Tot slot wordt er nog één stap gezet van 10 units naar 20 units. Gezien de vraag is een output van 20 units logisch. 

Mijn advies in deze zou zijn om de Dropout terug te brengen naar 0.2 en een logische verdeling te maken qua grote van units in de nn.Linear functie. Daarnaast, gezien de omvang van de data, is het ook verstandig om te kijken na een extra laag. Met het toevoegen van een extra laag kunnen de units beter verdeeld worden. In een handmatige test met de volgende configuratie is een nauwkeurigheid behaald van 0.7394
```
nn.Linear(config["13"], config["64"]),
nn.ReLU(),
nn.Linear(config["64"], config["32"]),
nn.Dropout(config["0.2"]),
nn.ReLU(),
nn.Linear(config["32"], config["20"]),
```


## 1b
Als je in de forward methode van het Linear model kijkt (in `tentamen/model.py`) dan kun je zien dat het eerste dat hij doet `x.mean(dim=1)` is. 
  * Het heeft wisselde lengte. Door dit te doen breng je het terug naar het gemiddelde van alle 13 (boven naar beneden kijken)

- Wat is het effect hiervan? Welk probleem probeert hij hier op te lossen? (maw, wat gaat er fout als hij dit niet doet?)
  * Effect: Neemt het gemiddelde van x over de dimension 1 (Mogelijk de aantal d te verkleinen), (mogelijk is plat maken ook een optie)
  * Welke probleem probeert hij op te lossen:
  * Wat gaat er fout als hij het niet doet?


- Hoe had hij dit ook kunnen oplossen?
  * Hoe had dit anders opgelost kunnen worden?
- Wat zijn voor een nadelen van de verschillende manieren om deze stap te doen?
  * Voor en nadelen

### 1c
Omdat jij de cursus Machine Learning hebt gevolgd kun jij hem uitstekend uitleggen wat een betere architectuur zou zijn.

- Beschrijf de architecturen die je kunt overwegen voor een probleem als dit. Het is voldoende als je beschrijft welke layers in welke combinaties je zou kunnen gebruiken.
  * Aangeven wat voor soort probleem het is. 
  * Hoeveel D's het heeft
  * layers in welke combinaties 

- Geef vervolgens een indicatie en motivatie voor het aantal units/filters/kernelsize etc voor elke laag die je gebruikt, en hoe je omgaat met overgangen (bv van 3 naar 2 dimensies). Een indicatie is bijvoorbeeld een educated guess voor een aantal units, plus een boven en ondergrens voor het aantal units. Met een motivatie laat je zien dat jouw keuze niet een random selectie is, maar dat je 1) andere problemen hebt gezien en dit probleem daartegen kunt afzetten en 2) een besef hebt van de consquenties van het kiezen van een range.
  * Motiveren van mijn antwoord.

- Geef aan wat jij verwacht dat de meest veelbelovende architectuur is, en waarom (opnieuw, laat zien dat je niet random getallen noemt, of keuzes maakt, maar dat jij je keuze baseert op ervaring die je hebt opgedaan met andere problemen).
  * architectuur wordt units/filyers/kernelsize enzo bedoeld???????????

### 1d
Implementeer jouw veelbelovende model: 

- Maak in `model.py` een nieuw nn.Module met jouw architectuur
  * Goed kijken of ik nog steeds een Linear model wil. 

- Maak in `settings.py` een nieuwe config voor jouw model
  * Past het bij de gekozen arcitectuur?

- Train het model met enkele educated guesses van parameters. 
  * Goed kijken welke het beste is. 

- Rapporteer je bevindingen. Ga hier niet te uitgebreid hypertunen (dat is vraag 2), maar rapporteer (met een afbeelding in `antwoorden/img` die je linkt naar jouw .md antwoord) voor bijvoorbeeld drie verschillende parametersets hoe de train/test loss curve verloopt.
  * Beantwoorden let op vergeet de afbeelding niet. 

- reflecteer op deze eerste verkenning van je model. Wat valt op, wat vind je interessant, wat had je niet verwacht, welk inzicht neem je mee naar de hypertuning.
  * Beantwoorden van de vraag en motivatie

Hieronder een voorbeeld hoe je een plaatje met caption zou kunnen invoegen.

<figure>
  <p align = "center">
    <img src="img/motivational.png" style="width:50%">
    <figcaption align="center">
      <b> Fig 1.Een motivational poster voor studenten Machine Learning (Stable Diffusion)</b>
    </figcaption>
  </p>
</figure>

## Vraag 2
Een andere collega heeft alvast een hypertuning opgezet in `dev/scripts/02_tune.py`.

### 2a
Implementeer de hypertuning voor jouw architectuur:
- zorg dat je model geschikt is voor hypertuning
  * In vraag 1 pas je waarschijnlijk dingen aan, dan hier ook aanpassen. 

- je mag je model nog wat aanpassen, als vraag 1d daar aanleiding toe geeft. Als je in 1d een ander model gebruikt dan hier, geef je model dan een andere naam zodat ik ze naast elkaar kan zien.
  * Duidelijk

- (Opmerking) Stel dat je nog wat wilt aanpassen, wat zou je dan aanpassen? (vraag was niet volledig)
- voeg jouw model in op de juiste plek in de `tune.py` file.
  * Verwijzen naar het juiste model??????????

- maak een zoekruimte aan met behulp van pydantic (naar het voorbeeld van LinearSearchSpace), maar pas het aan voor jouw model.
  * Zoek ruimte aanpassen let op als ik daar dingen aanpas, ik dat ook doe op andere plekken ivm namen

- Licht je keuzes toe: wat hypertune je, en wat niet? Waarom? En in welke ranges zoek je, en waarom? Zie ook de [docs van ray over search space](https://docs.ray.io/en/latest/tune/api_docs/search_space.html#tune-sample-docs) en voor [rondom search algoritmes](https://docs.ray.io/en/latest/tune/api_docs/suggestion.html#bohb-tune-search-bohb-tunebohb) voor meer opties en voorbeelden.
  * Beantwoorden en argumenteren van antwoord. 

### 2b
- Analyseer de resultaten van jouw hypertuning; visualiseer de parameters van jouw hypertuning en sla het resultaat van die visualisatie op in `reports/img`. Suggesties: `parallel_coordinates` kan handig zijn, maar een goed gekozen histogram of scatterplot met goede kleuren is in sommige situaties duidelijker! Denk aan x en y labels, een titel en units voor de assen.
  * Analyseren
  * Visualiseren 
  * less = more!

- reflecteer op de hypertuning. Wat werkt wel, wat werkt niet, wat vind je verrassend, wat zijn trade-offs die je ziet in de hypertuning, wat zijn afwegingen bij het kiezen van een uiteindelijke hyperparametersetting.
  * Argumenteren

Importeer de afbeeldingen in jouw antwoorden, reflecteer op je experiment, en geef een interpretatie en toelichting op wat je ziet.
 * 

### 2c
- Zorg dat jouw prijswinnende settings in een config komen te staan in `settings.py`, en train daarmee een model met een optimaal aantal epochs, daarvoor kun je `01_model_design.py` kopieren en hernoemen naar `2c_model_design.py`.

## Vraag 3
### 3a
- fork deze repository.
- Zorg voor nette code. Als je nu `make format && make lint` runt, zie je dat alles ok is. Hoewel het in sommige gevallen prima is om een ignore toe te voegen, is de bedoeling dat je zorgt dat je code zoveel als mogelijk de richtlijnen volgt van de linters.
- We werken sinds 22 november met git, en ik heb een `git crash coruse.pdf` gedeeld in les 2. Laat zien dat je in git kunt werken, door een git repo aan te maken en jouw code daarheen te pushen. Volg de vuistregel dat je 1) vaak (ruwweg elke dertig minuten aan code) commits doet 2) kleine, logische chunks van code/files samenvoegt in een commit 3) geef duidelijke beschrijvende namen voor je commit messages
- Zorg voor duidelijke illustraties; voeg labels in voor x en y as, zorg voor eenheden op de assen, een titel, en als dat niet gaat (bv omdat het uit tensorboard komt) zorg dan voor een duidelijke caption van de afbeelding waar dat wel wordt uitgelegd.
- Laat zien dat je je vragen kort en bondig kunt beantwoorden. De antwoordstrategie "ik schiet met hagel en hoop dat het goede antwoord ertussen zit" levert minder punten op dan een kort antwoord waar je de essentie weet te vangen. 
- nodig mij uit (github handle: raoulg) voor je repository. 
