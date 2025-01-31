# Tentamen ML2022-2023

De opdracht is om de audio van 10 cijfers, uitgesproken door zowel mannen als vrouwen, te classificeren. De dataset bevat timeseries met een wisselende lengte.

In [references/documentation.html](references/documentation.html) lees je o.a. dat elke timestep 13 features heeft.
Jouw junior collega heeft een neuraal netwerk gebouwd, maar het lukt hem niet om de accuracy boven de 67% te krijgen. Aangezien jij de cursus Machine Learning bijna succesvol hebt afgerond hoopt hij dat jij een paar betere ideeen hebt.

## Vraag 1

### 1a
In `dev/scripts` vind je de file `01_model_design.py`.
Het model in deze file heeft in de eerste hidden layer 100 units, in de tweede layer 10 units, dit heeft jouw collega ergens op stack overflow gevonden en hij had gelezen dat dit een goed model zou zijn.
De dropout staat op 0.5, hij heeft in een blog gelezen dat dit de beste settings voor dropout zou zijn.



<div style="border-radius: 10px; background: ghostwhite; padding: 10px;">
 &#10002; Vraag 1a (deel 1)
<br>
Wat vind je van de architectuur die hij heeft uitgekozen (een Neuraal netwerk met drie Linear layers)? Wat zijn sterke en zwakke kanten van een model als dit in het algemeen? En voor dit specifieke probleem?
</div>

<br>

<div style="border-radius: 10px; background: ghostwhite; padding: 10px;">
 &#9997; Antwoord 1a (deel 1)
</div>
Het gekozen Neural Network met Linear Layers is een relatief simpel model. Het model bestaat uit de volgende elementen: 
Input -> Linear Layer 1 -> ReLU -> Linear Layer 2 -> Dropout -> ReLU -> Lenear Layer 3 -> Output. 
Vanwege de (relatieve) eenvoud van het model, helpt dit om overfitting te vermijden. Wegens de simpliciteit en snelheid is het een goed basismodel om mee te starten. Tegelijk is dit ook direct een groot nadeel van dit model. Omdat het een (algemeen) simpel model is behaald het niet (altijd) de hoogt mogelijke nauwkeurigheid. In relatie tot de data en de vraag zal er dus gekeken moeten worden naar een meer specifiek model om een hogere nauwkeurigheid te behalen. Voor dit specifieke probleem, zijnde classificatie van audio, is een model zoals deze niet (volledig) passend. Om een hogere nauwkeurigheid te behalen kan er gekeken worden naar convolutional neural networks (CNNs) of misschien zelfs beter: Recurrent Neural Networks (RNN). RNN zijn specifiek goed in sequentiële gegevens zoals tekst, audio en video.


<br>

<div style="border-radius: 10px; background: ghostwhite; padding: 10px;">
 &#10002; Vraag 1a (deel 2)
<br>
Wat vind je van de keuzes die hij heeft gemaakt in de LinearConfig voor het aantal units ten opzichte van de data? En van de dropout?
</div>

<br>

<div style="border-radius: 10px; background: ghostwhite; padding: 10px;">
 &#9997; Antwoord 1a (deel 2)
</div>

De vraag vanuit de betreffende collega is om de data te classificeren. De data bestaat uit getalen van nul tot negen (_n_=10), uitgesproken in het Arabic door mannelijke (_n_=44) en vrouwelijke sprekers (_n_=44). Dit resulteert in totaal 20 classes die geïdentificeerd dienen te worden. Kijkend naar het model geschreven door de collega zien we het volgende:
```
(Getalen overgenomen om het leesbaar te maken)
nn.Linear(config["13"], config["100"]),
nn.ReLU(),
nn.Linear(config["100"], config["10"]),
nn.Dropout(config["0.5"]),
nn.ReLU(),
nn.Linear(config["10"], config["20"]),
```
De stappen in het model zijn (relatief) groot. Van 13 units (aantal attributen) naar 100 units is een relatief grote stap. Vervolgens wordt er een rectified linear unit (ReLU) toegepast als activatie functie (f(x)=max(0,x). De volgende stap is ook relatief groot namelijk: van 100 units naar 10 units. Aansluitend wordt er een Dropout functie toegepast om overfitting te voorkomen. De Dropout staat ingesteld op 0.5. Dit is een relatief hoge waarde wat resulteert in het feit dat de helft van de neurons worden uitgeschakeld. Vanuit de literatuur wordt geadviseerd om de Dropout in te stellen tussen de 0.2 en 0.5. Na de Dropout wordt er opnieuw een ReLU toegepast. Tot slot wordt er nog één stap gezet van 10 units naar 20 units. Gezien de vraag is een output van 20 logisch. De stap van 10 units terug naar 20 units is minder logisch. 
Mijn advies aan de college zou zijn: 
-	Dropout terug brengen naar 0.2.
-	Logische verdeling maken qua grote van units in de nn.Linear functie.
-	Gezien omvang van de data, kijken of er een extra laag toegevoegd kan worden. 
Met het toevoegen van een extra laag kunnen de units beter verdeeld worden. In een handmatige test met de volgende configuratie (zie hieronder) is een nauwkeurigheid behaald van 0.7394 (zonder extra laag) en 0.7470 (met extra laag)

_Dropout 0.2, logische verdeling, zonder extra laag_
```
nn.Linear(config["13"], config["64"]),
nn.ReLU(),
nn.Linear(config["64"], config["32"]),
nn.Dropout(config["0.2"]),
nn.ReLU(),
nn.Linear(config["32"], config["20"]),
```
_Dropout 0.2, logische verdeling, met extra laag_
```
nn.Linear(config["13"], config["64"]),
nn.ReLU(),
nn.Linear(config["64"], config["128"]),
nn.Dropout(config["0.2"]),
nn.Linear(config["128"], config["32"]),
nn.Dropout(config["0.2"]),
nn.ReLU(),
nn.Linear(config["32"], config["20"]),
```

## 1b
Als je in de forward methode van het Linear model kijkt (in `tentamen/model.py`) dan kun je zien dat het eerste dat hij doet `x.mean(dim=1)` is.

<div style="border-radius: 10px; background: ghostwhite; padding: 10px;">
 &#10002; Vraag 1b (deel 1)
<br>
Wat is het effect hiervan? Welk probleem probeert hij hier op te lossen? (maw, wat gaat er fout als hij dit niet doet?)
</div>

<br>

<div style="border-radius: 10px; background: ghostwhite; padding: 10px;">
 &#9997; Antwoord 1b (deel 1)
</div>



De complete functie is
```
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=1)
        x = self.encoder(x)
        return x
```
Het deel x.mean(dim=1) pakt het gemiddelde van alle regels in een block. Uitkomst is één regel met 13 getallen (13 attributen) (e.g. -2.5929 -2.889 0.29554 -0.067409 0.28635 0.20898 0.41408 0.38878 0.37271 0.16329 0.0050341 0.12431 0.44326). Deze oplossing lost het probleem op van de variabele lengte van de blocks. Deze stap is nodig omdat et gekozen model niet overweg kan met verschillende block lengtes. 



<div style="border-radius: 10px; background: ghostwhite; padding: 10px;">
 &#10002; Vraag 1b (deel 2)
<br>
Hoe had hij dit ook kunnen oplossen?
</div>

<br>

<div style="border-radius: 10px; background: ghostwhite; padding: 10px;">
 &#9997; Antwoord 1b (deel 2)
</div>

Andere opties zijn:
nn.Flatten(), nn.AvgPool2d(), nn.MaxPool2d()


<div style="border-radius: 10px; background: ghostwhite; padding: 10px;">
 &#10002; Vraag 1b (deel 3)
<br>
Wat zijn voor een nadelen van de verschillende manieren om deze stap te doen?
</div>

<br>

<div style="border-radius: 10px; background: ghostwhite; padding: 10px;">
 &#9997; Antwoord 1b (deel 3)
</div>


* nn.Flatten(): hervormd de data naar een 1-dim array
  * Nadeel: Kost veel geheugen en processing kracht. Daarnaast zorgt deze methoden er ook voor dat je (ruimtelijke) informatie verliest.
* nn.AvgPool2d(): pakt het gemiddelde van de gekozen window.
  * Nadeel: door het gemiddelde te pakken kan je belangrijke elementen uit de data kwijtraken. Daarnaast hebben outliers een groot effect op het gemiddelde. 
* nn.MaxPool2d(): pakt de mix waarde van de gekozen window. 
  * Nadeel: ook deze methoden kan leiden tot het verlies van informatie. Dit komt omdat alleen de max waarde van elke window wordt meegenomen. Door deze methoden gaan kleine details verloren. 


### 1c
Omdat jij de cursus Machine Learning hebt gevolgd kun jij hem uitstekend uitleggen wat een betere architectuur zou zijn.



<div style="border-radius: 10px; background: ghostwhite; padding: 10px;">
 &#10002; Vraag 1c (deel 1)
<br>
Beschrijf de architecturen die je kunt overwegen voor een probleem als dit. Het is voldoende als je beschrijft welke layers in welke combinaties je zou kunnen gebruiken.
</div>

<br>

<div style="border-radius: 10px; background: ghostwhite; padding: 10px;">
 &#9997; Antwoord 1c (deel 1)
</div>


Zoals eerder beschreven gaat het om een classificatie vraag. De dataset bestaat uit 8800 (10 cijfers x 10 herhalingen x 88 sprekers) tijdreeksen van 13 MFCCs. Na verkenning komende de volgende architecturen als beste naar boven gezien de vraag en gegeven dataset:


<figure>
  <p align = "center">
    <img src="img/RNN GRU LSTM.png" style="width:100%">
    <figcaption align="center">
      <b> Fig 1. RNN, GRU en LSTM</b>
    </figcaption>
  </p>
</figure>

* RNN
  * Goed voor tekst, audio of tijdreeksgegevens.
  * Heeft een “hidden state” zodat het netwerk informatie van eerdere elementen onthoud. 
  * Zeer geschrikt voor sequentie data.
* LSTM
  * Type RNN maar beter in het vasthouden van langetermijnrelaties in de data. 
  * Complexere structuur dan RNN zijnde: Input, output en forget gates.
  * Vaak gebruikt voor vertaal modellen en taalmodellering 
* GRU
  * Type RNN maar met een eenvoudiger structuur dan LSTM.
  * Gemakkelijk te trainen (i.v.m. LSTM).
  * Twee poorten: Update gate en een reset gate

Omdat we hier te maken hebben met sequentie data is het van belang om de context te bewaren uit het verleden (RNN grootste probleem is vanishing gradient problem). LSTM en GRU kunnen dat beide. Gezien de data en het vraagstuk valt mijn keuze op het maken van een GRU-model. Deze is makkelijker te trainen, minder complex maar biedt vrijwel dezelfde mogelijkheden. 

<div style="border-radius: 10px; background: ghostwhite; padding: 10px;">
 &#10002; Vraag 1c (deel 2 en 3)
<br>

- Geef vervolgens een indicatie en motivatie voor het aantal units/filters/kernelsize etc voor elke laag die je gebruikt, en hoe je omgaat met overgangen (bv van 3 naar 2 dimensies). Een indicatie is bijvoorbeeld een educated guess voor een aantal units, plus een boven en ondergrens voor het aantal units. Met een motivatie laat je zien dat jouw keuze niet een random selectie is, maar dat je 1) andere problemen hebt gezien en dit probleem daartegen kunt afzetten en 2) een besef hebt van de consquenties van het kiezen van een range.
- Geef aan wat jij verwacht dat de meest veelbelovende architectuur is, en waarom (opnieuw, laat zien dat je niet random getallen noemt, of keuzes maakt, maar dat jij je keuze baseert op ervaring die je hebt opgedaan met andere problemen).
</div>

<br>

<div style="border-radius: 10px; background: ghostwhite; padding: 10px;">
 &#9997; Antwoord 1c (deel 2 en 3 (deels ook al beantwoord in deel 1))
</div>

Voor het maken van een GRU-model zijn de volgende gegevens nodig (zie hieronder). Achter elk antwoord direct de overweging beschreven. 

|Name|Option|Reasoning|
|----|---|---|
|Input|13|I.v.m. de aantal features (aantal MFCC coefficients)|
|Hidden size|64|Mijn gevoel is dat dit een goed startpunt is. Dit kan nog veranderen naar 128 of zelfs 256, afhankelijk van de resultaten uit het experiment.|
|Output|20|Dit komt overeen met de aantal classes die geïdentificeerd moeten worden.|
|Number of layers|2 of 4|Op die manier zijn er voldoende lagen om het model te kunnen trainen.|
|Loss function|Cross-Entropy-Loss|Zoals besproken tijdens het college is dit het best passend bij classificatie.|
|Learning rate|0,001|Op basis van eerder uitgevoerde experimenten tijdens het college lijkt mij dit de beste keuze.|
|Optimizer|Adam|Deze neemt informatie mee (past gradients) om de learning rate aan te passen. Dit is het best passend voor dit vraagstuk.|



### 1d

<div style="border-radius: 10px; background: ghostwhite; padding: 10px;">
 &#10002; Vraag 1d
<br>

Implementeer jouw veelbelovende model: 
- Maak in `model.py` een nieuw nn.Module met jouw architectuur 
- Maak in `settings.py` een nieuwe config voor jouw model
- Train het model met enkele educated guesses van parameters. 
- Rapporteer je bevindingen. Ga hier niet te uitgebreid hypertunen (dat is vraag 2), maar rapporteer (met een afbeelding in `antwoorden/img` die je linkt naar jouw .md antwoord) voor bijvoorbeeld drie verschillende parametersets hoe de train/test loss curve verloopt.
- reflecteer op deze eerste verkenning van je model. Wat valt op, wat vind je interessant, wat had je niet verwacht, welk inzicht neem je mee naar de hypertuning.
</div>

<br>

<div style="border-radius: 10px; background: ghostwhite; padding: 10px;">
 &#9997; Antwoord 1d
</div>


Omdat de input, output en de dropout al redelijk vaststaan heb ik hier alleen geëxperimenteerd met de hidden size en de num_layers. De optimizer is de beste keuze voor dit vraagstuk en expirimenteren met de learning_rate zou te veel gaan lijken op hypertuning. 
Het meest opvallende is dat dit model al vrij snel boven de 0.9300 komt. Een relatief groot effect heeft het aanpassen van de hidden size. De aanpassing van de num_layers heeft een minder groot effect dan verwacht. 


| Score |0.9389|0.9439|0.9632|0.9582|0.9605|0.9366|
|----|---|---|---|---|---|---|
|Data|20230129-1244|20230129-1250|20230129-1300|20230129-1311|20230129-1334|20230129-1344|
|input|13|13|13|13|13|13|
|output|20|20|20|20|20|20|
|hidden|64|128|256|256|128|64|
|dropout|0.2|0.2|0.2|0.2|0.2|0.2|
|num_layers|2|2|2|4|4|4|
|optimizer|Adam|Adam|Adam|Adam|Adam|Adam|
|learning_rate|0.001|0.001|0.001|0.001|0.001|0.001|

In de grafiek hieronder is te zien dat de eerste en laatste run beide geleidelijk dalen. Waar de andere vier runs vrij snel dalen voor dat ze afvlakken. Deze vier runs stoppen rond de 12 – 14 epoch met leren. De andere twee lijken door te gaan met leren nog tot in de 22ste epoch.

<figure>
  <p align = "center">
    <img src="img/Loss train.png" style="width:100%">
    <figcaption align="center">
      <b> Fig 2. Lost train</b>
    </figcaption>
  </p>
</figure>

In deze grafiek hieronder zien we hetzelfde als in de grafiek hierboven. De eerste en laatste run zijn gelijk en de andere vier zijn gelijk aan elkaar. 

<figure>
  <p align = "center">
    <img src="img/Loss test.png" style="width:100%">
    <figcaption align="center">
      <b> Fig 3. Lost test</b>
    </figcaption>
  </p>
</figure>

## Vraag 2
Een andere collega heeft alvast een hypertuning opgezet in `dev/scripts/02_tune.py`.

### 2a


<div style="border-radius: 10px; background: ghostwhite; padding: 10px;">
 &#10002; Vraag 2a
<br>

Implementeer de hypertuning voor jouw architectuur:
- zorg dat je model geschikt is voor hypertuning
- je mag je model nog wat aanpassen, als vraag 1d daar aanleiding toe geeft. Als je in 1d een ander model gebruikt dan hier, geef je model dan een andere naam zodat ik ze naast elkaar kan zien.
- voeg jouw model in op de juiste plek in de `tune.py` file.
- maak een zoekruimte aan met behulp van pydantic (naar het voorbeeld van LinearSearchSpace), maar pas het aan voor jouw model.
- Licht je keuzes toe: wat hypertune je, en wat niet? Waarom? En in welke ranges zoek je, en waarom? Zie ook de [docs van ray over search space](https://docs.ray.io/en/latest/tune/api_docs/search_space.html#tune-sample-docs) en voor [rondom search algoritmes](https://docs.ray.io/en/latest/tune/api_docs/suggestion.html#bohb-tune-search-bohb-tunebohb) voor meer opties en voorbeelden.
</div>

<br>

<div style="border-radius: 10px; background: ghostwhite; padding: 10px;">
 &#9997; Antwoord 2a
</div>


Vanuit vraag 1 is naar voren gekomen dat een hidden van 128, Dropout van 0.2 en een Num_layers van 4 tot nu toe het beste resultaat heeft gegeven. Vanuit een search online is naar voren gekomen dat het ook verstandig is om de batchsize mee te nemen in het hypertunen. Daarom deze ook meegenomen in de SearchSpace. 

Voor de eerste run de SearchSpace ingesteld met enige ruimte rond de parameters die in vraag 1 het beste resultaat hebben opgeleverd. 

```
class grumodelSearchSpace(BaseSearchSpace):
    hidden: Union[int, SAMPLE_INT] = tune.randint(64, 256)
    dropout: Union[float, SAMPLE_FLOAT] = tune.uniform(0.1, 0.3)
    num_layers: Union[int, SAMPLE_INT] = tune.randint(2, 6)
    batchsize: Union[int, SAMPLE_INT] = tune.randint(16, 256)
```



### 2b

<div style="border-radius: 10px; background: ghostwhite; padding: 10px;">
 &#10002; Vraag 2b
<br>

- Analyseer de resultaten van jouw hypertuning; visualiseer de parameters van jouw hypertuning en sla het resultaat van die visualisatie op in `reports/img`. Suggesties: `parallel_coordinates` kan handig zijn, maar een goed gekozen histogram of scatterplot met goede kleuren is in sommige situaties duidelijker! Denk aan x en y labels, een titel en units voor de assen.
- reflecteer op de hypertuning. Wat werkt wel, wat werkt niet, wat vind je verrassend, wat zijn trade-offs die je ziet in de hypertuning, wat zijn afwegingen bij het kiezen van een uiteindelijke hyperparametersetting.
Importeer de afbeeldingen in jouw antwoorden, reflecteer op je experiment, en geef een interpretatie en toelichting op wat je ziet.
</div>

<br>

<div style="border-radius: 10px; background: ghostwhite; padding: 10px;">
 &#9997; Antwoord 2b
</div>


Vanuit vraag 1 neem ik het GRU model mee omdat ik daar al een nauwkeurigheid heb weten te behalen van zo’n 96%. Met de volgende parameters (uitgelegd in vraag 2a) ben ik gestart met het hypertunen:
### Run 1
|Subject|Between| 
|---|---|
|Hidden|64, 256|
|Num_layers|2, 6|
|Dropout|0.1, 0.3|
|Batchsize|16, 256|
|Epochs|20|

<figure>
  <p align = "center">
    <img src="img/RUN1GRA.png" style="width:100%">
    <figcaption align="center">
      <b> Fig 4. Grafiek run 1</b>
    </figcaption>
  </p>
</figure>

In bovenstaande parallel coordinates plot (fig. 4) is te zien dat de hoogst behaald resultaat de volgende instellingen heeft: batchsize: 252, dropout: 0.16136, hidden: 171 en num_layers: 5. In fig. 5 wordt dit in een tabel weergegeven. 

<figure>
  <p align = "center">
    <img src="img/RUN1TAB.png" style="width:100%">
    <figcaption align="center">
      <b> Fig 5. Best score run 1</b>
    </figcaption>
  </p>
</figure>

In onderstaande scatter plot (Fig. 6) worden de zelfde resultaten weergegeven als in de fig. 4. Vanuit fig. 6 (batchsize) is op te maken dat een hoge batchsize niet perse resulteert in een hogere nauwkeurigheid. Batchsize van 220+ hebben ook resultaten behaald van minder dan 0.6 nauwkeurigheid. Dit zelfde fenomeen is ook terug te zien in fig. 6 (dropout). Hier is het ook zo dat zowel een hoge als lage dropout voor zowel een hoge als lage nauwkeurigheid heeft gezorgd. Kijkend naar fig. 6 (hidden) is te zien dat de nauwkeurigheid het hoogst is wanneer de waarde ligt tussen de 160 en 180. In vergelijking met de voorgaande (batchsize, dropout) is hier meer groepsvorming te zien. In fig 6. (num_layers) is ook groepsvorming te zien bij een hoge nauwkeurigheid tussen de 4.0 en 5.0. Tegelijk is ook te zien dat er ook een lage nauwkeurigheid behaald is met deze aantal lagen. 

<figure>
  <p align = "center">
    <img src="img/RUN1SCATTER.png" style="width:100%">
    <figcaption align="center">
      <b> Fig 6. Scatterplot run 1</b>
    </figcaption>
  </p>
</figure>

Kijkend naar de spreiding per onderdeel is het lastig om te bepalen of dit nu het beste resultaat heeft opgeleverd. De nauwkeurigheid is immers maar iets hoger dan tijdens de handmatige fase (vraag 1d). Om te zien of er toch een beter resultaat te behalen valt heb ik nog een run gedraaid (run 2). In deze run besloten om de hidden meer te centreren rond de waarde waar het hoogte resultaat mee is behaald zijnde 171. Daarom de range ingesteld op 128, 256. Omdat de Num_layers hoogste resultaat heeft behaald met 5 deze verhoogt naar range 4, 8. Omdat de dropout een hoge nauwkeurigheid heeft behaald met zowel een lage als een hoge dropout heb ik deze niet veranderd. Omdat een lager batchsize alleen lage nauwkeurigheid heeft behaald als resultaat deze verhoogt om te zien of dit een positief effect zal hebben (range naar 256, 512).


Run 2
|Subject|Between| 
|---|---|
|Hidden|128, 256|
|Num_layers|4, 8|
|Dropout|0.1, 0.3|
|Batchsize|256, 512|
|Epochs|20|

<figure>
  <p align = "center">
    <img src="img/RUN2GRA.png" style="width:100%">
    <figcaption align="center">
      <b> Fig 7. Grafiek run 2</b>
    </figcaption>
  </p>
</figure>

In fig. 7 is te zien dat een hogere batch geen (positief) effect heeft gehad op de nauwkeurigheid van het model. Dit is ook terug te zien in het tabel, zie fig. 8. Hierin is terug te zien dat de hoogte nauwkeurigheid zijnde 0.94347 lager uitvalt dan in de eerste run. 

<figure>
  <p align = "center">
    <img src="img/RUN2TAB.png" style="width:100%">
    <figcaption align="center">
      <b> Fig 8. Best score run 2</b>
    </figcaption>
  </p>
</figure>

In fig. 9 wordt run 2 weergegeven in gesepareerde scatter plots. Uit fig. 9 (batchsize) is op te maken dat een batchsize van rond de 450 resulteert in de hoogste nauwkeurigheid voor run 2. Voor de dorpout is het resultaat wat meer verspreid zo ook voor de num_layers al valt wel op dat een hoge num_layers (7) heeft meegedragen aan een hoge nauwkeurigheid. In fig. 9 (hidden) is te zien dat er enige groepsvorming is rond 160 voor een hogere nauwkeurigheid. 

<figure>
  <p align = "center">
    <img src="img/RUN2SCATTER.png" style="width:100%">
    <figcaption align="center">
      <b> Fig 9. Scatterplot run 2</b>
    </figcaption>
  </p>
</figure>


Run 3

Om er zeker van te zijn dat ik niet de verkeerde kant op aan het zoeken ben heb ik nog een derde run uitgevoerd. In deze run heb ik Num_layers en Batchsize juist verlaagt, zie tabel hieronder. 

|Subject|Between| 
|---|---|
|Hidden|128, 256|
|Num_layers|1, 3|
|Dropout|0.1, 0.3|
|Batchsize|16, 32|
|Epochs|20|

In fig. 10 is terug te zien dat de nauwkeurigheid score wederom boven de 0.9 is uitgekomen. In fig. 11 is af te lezen dat het nauwkeurigheid score is uitgekomen op 0.94072. 

<figure>
  <p align = "center">
    <img src="img/RUN3GRA.png" style="width:100%">
    <figcaption align="center">
      <b> Fig 10. Grafiek run 3</b>
    </figcaption>
  </p>
</figure>

<figure>
  <p align = "center">
    <img src="img/RUN3TAB.png" style="width:100%">
    <figcaption align="center">
      <b> Fig 11. Best score run 3</b>
    </figcaption>
  </p>
</figure>

In fig. 12 hieronder worden de afzonderlijke scores gesepareerde weergegeven in een scatter plot. Vanuit fig. 12 is op te maken dat ook een lage batchsize kan resulteren in een relatief hoge nauwkeurigheid. Daarnaast is te zien dat de dorpout en de hidden beiden relatief dezelfde plaatsing innemen als in run 2. De num_layers geeft hier ook weer de voorkeur aan een hogere waarde. 

<figure>
  <p align = "center">
    <img src="img/RUN3SCATTER.png" style="width:100%">
    <figcaption align="center">
      <b> Fig 12. Scatterplot run 3</b>
    </figcaption>
  </p>
</figure>

Om een volledig overzicht te creëren wordt in fig. 13 alle parallel coordinates plot onder elkaar geplaatst. 

<figure>
  <p align = "center">
    <img src="img/ALL3.png" style="width:100%">
    <figcaption align="center">
      <b> Fig 13. Parallel coordinates plots RUN 1, 2 en 3</b>
    </figcaption>
  </p>
</figure>

In fig. 13 is te zien dat de rode lijn in alle drie de plots enigszins op dezelfde wijze loopt. Dit geeft het idee dat er een relatie is tussen deze parameters. De batchsize bepaald de aantal samples die in één passage wordt gebruikt. Een grotere batchsize resulteert vaak in een stabielere gradient. Dit vraagt wel meer geheugenruimte. Een kleine batchsize vraagt minder geheugenruimte maar kan resulteren in een ruisachtiger gradient. De dropout is een regularisatietechniek die overfitting helpt voorkomen. Deze techniek werkt door neuronen in het netwerk (tijdens de training) willekeurig te laten vallen. De enige relatie tussen deze twee parameters is dat ze beide het generalisatievermogen van het model beïnvloeden. Hidden kan lange termijn afhankelijkheden tussen de inputs vastleggen (wat nuttig is bij sequentiële gegevens). De dropout heeft invloed op de hidden vanwege de regulerende functie die de dropout heeft. Tot slot kunnen we nog kijken naar de relatie tussen hidden en num_layers. Hierbij zien we dat de hidden de capaciteit van elke GRU-laag bepaald en de num_layers de diepte van het model bepaalt. Uit de literatuur is naar voren gekomen dat een model met veel hidden en veel num_layers niet per definitie beter is dan een model met weinig hidden en weinig num_layers. Gezien de beschikbare ruimte en tijd besluit ik om als prijswinnende setting het resultaat van RUN 1 te gebruiken. Dit wilt niet zeggen dat ik denk dat ik geen hoger resultaat kan behalen. Na onderzoek online denk ik dat ik een hoger resultaat kan behalen door een attention layer toe te voegen aan mijn GRU-model. Een attention layer stelt het GRU-model in staat zijn aandacht te richten op verschillende delen van de input, i.v.m. de gehele input te gebruiken. De attention layer kent gewichten toe aan elke tijdstap van de input om aan te geven hoe belangrijk die is voor de voorspelling van de output. De attention layer helpt het model om alleen op de meest relevante delen van de input te focussen. 


### 2c

<div style="border-radius: 10px; background: ghostwhite; padding: 10px;">
 &#10002; Vraag 2c
<br>

- Zorg dat jouw prijswinnende settings in een config komen te staan in `settings.py`, en train daarmee een model met een optimaal aantal epochs, daarvoor kun je `01_model_design.py` kopieren en hernoemen naar `2c_model_design.py`.
</div>

<br>

<div style="border-radius: 10px; background: ghostwhite; padding: 10px;">
 &#9997; Antwoord 2c
</div>

Om te achterhalen wat de prijswinnende settings zijn heb ik de waardes uit RUN 1 gebruikt om 2c_model_design.py te maken. Vervolgens heb ik gekeken wanneer het beste resultaat behaald wordt door het model 150 epochs te laten draaien, zie fig. 14.



<figure>
  <p align = "center">
    <img src="img/model_design.png" style="width:100%">
    <figcaption align="center">
      <b> Fig 14. Accuracy extra run</b>
    </figcaption>
  </p>
</figure>

In fig. 14 is af te lezen dat het beste resultaat behaald is tijdens de 56ste epoch. In fig. 15 (hieronder) is te zien, aan de loss (test) ratio, dat het beste model rond de 51ste epoch bevindt. 

<figure>
  <p align = "center">
    <img src="img/model_design_loss_test.png" style="width:100%">
    <figcaption align="center">
      <b> Fig 15. Loss/test extra run</b>
    </figcaption>
  </p>
</figure>

In fig. 16 (hieronder) is aan de hand van de loss (train) ratio te zien dat het model na ongeveer de 35ste epoch niet meer leert met uitzondering van twee kleine golfjes rond de 50ste epoch en rond de 58ste epoch. 

<figure>
  <p align = "center">
    <img src="img/model_design_loss_train.png" style="width:100%">
    <figcaption align="center">
      <b> Fig 16. Loss/train extra run</b>
    </figcaption>
  </p>
</figure>

Na aanleiding van bovengenoemde resultaten en met namen het resultaat in fig. 14 besloten om 56 epochs aan te houden als beste instelling voor mijn model. Om er zeker van te zijn dat dit het beste model is, deze nogmaals gedraaid met de prijswinnende instellingen, zie fig. 17 en 18.

<figure>
  <p align = "center">
    <img src="img/model_design_FINAL.png" style="width:100%">
    <figcaption align="center">
      <b> Fig 17. Loss/test, Loss/train en learning rate beste model</b>
    </figcaption>
  </p>
</figure>

<figure>
  <p align = "center">
    <img src="img/model_design_FINAL_Graph.png" style="width:100%">
    <figcaption align="center">
      <b> Fig 18. Nauwkeurigheid beste model</b>
    </figcaption>
  </p>
</figure>

Vanuit fig. 18 is op te maken dat met een resultaat van 0.9633 dit inderdaad de prijswinnende instellingen zijn. Het verschil met de laatste run uit vraag 1d is verwaarloosbaar. In fig. 17 (Loss/test) is af te lezen dat het laagste punt bereid wordt aan het eind van de gekozen aantal epochs. Dit geeft aan dat het model rond de gekozen aantal epochs het beste scored op niet eerder gebruikte data. In fig 17 (Loss/train) is af te lezen dat, met deze instellingen, vanaf halverwege er niet veel meer geleerd wordt. Tegelijk is te zien dat aan het eind er toch nog wat fluctuaties te zien zijn. Dit komt redelijk overeen met wat in fig. 17 (learning_rate) af te lezen is. 

## Conclusie

Zoals eerder aangegeven is het beste resultaat zonder het toevoegen van een attention layer rond de 0.96XX. Gezien de vraag en de dataset is mijn persoonlijke inschatting dat een foutmarge van 0.04 acceptabel is. Mocht de vraag zijn om meer specifiek inzicht te krijgen in de nauwkeurigheid van het model is het verstandig om een analyse te doen aan de hand van een confusion matrix. 



## Vraag 3

<div style="border-radius: 10px; background: ghostwhite; padding: 10px;">
 &#10002; Vraag 3a
<br>

- fork deze repository.
- Zorg voor nette code. Als je nu `make format && make lint` runt, zie je dat alles ok is. Hoewel het in sommige gevallen prima is om een ignore toe te voegen, is de bedoeling dat je zorgt dat je code zoveel als mogelijk de richtlijnen volgt van de linters.
- We werken sinds 22 november met git, en ik heb een `git crash coruse.pdf` gedeeld in les 2. Laat zien dat je in git kunt werken, door een git repo aan te maken en jouw code daarheen te pushen. Volg de vuistregel dat je 1) vaak (ruwweg elke dertig minuten aan code) commits doet 2) kleine, logische chunks van code/files samenvoegt in een commit 3) geef duidelijke beschrijvende namen voor je commit messages
- Zorg voor duidelijke illustraties; voeg labels in voor x en y as, zorg voor eenheden op de assen, een titel, en als dat niet gaat (bv omdat het uit tensorboard komt) zorg dan voor een duidelijke caption van de afbeelding waar dat wel wordt uitgelegd.
- Laat zien dat je je vragen kort en bondig kunt beantwoorden. De antwoordstrategie "ik schiet met hagel en hoop dat het goede antwoord ertussen zit" levert minder punten op dan een kort antwoord waar je de essentie weet te vangen. 
- nodig mij uit (github handle: raoulg) voor je repository. 
</div>

<br>

<div style="border-radius: 10px; background: ghostwhite; padding: 10px;">
 &#9997; Antwoord 3a
</div>

Het resultaat van _make format && make lint_:
```
make format && make lint
poetry run isort dev
Fixing /home/azureuser/Tentamen/ML22-tentamen/dev/scripts/02_tune.py
poetry run black dev
reformatted dev/scripts/2c_model_design.py
reformatted dev/scripts/01_gru_model.py
reformatted dev/scripts/02_tune.py

All done! ✨ 🍰 ✨
3 files reformatted, 2 files left unchanged.
poetry run isort tentamen
poetry run black tentamen
reformatted tentamen/settings.py
reformatted tentamen/model.py

All done! ✨ 🍰 ✨
2 files reformatted, 6 files left unchanged.
poetry run flake8 dev
poetry run flake8 tentamen
tentamen/model.py:50:8: N801 class name 'grumodel' should use CapWords convention
tentamen/settings.py:60:8: N801 class name 'gru_modelConfig' should use CapWords convention
tentamen/settings.py:69:8: N801 class name 'grumodelSearchSpace' should use CapWords convention
```
Bovenstaande aangepast naar CapWords convention.

---

<figure>
  <p align = "center">
    <img src="img/Banner.png" style="width:100%">
  </p>
</figure>