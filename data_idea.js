// Remplace tous les "Brouillon" par "Validé" dans ma feuille.
// Je veux changer tous les "Non payé" en "Payé" dans la feuille actuelle.
// Peux-tu écrire une fonction qui remplace toutes les occurrences de "En attente" par "Traitée" ?
// Modifie toutes les cellules contenant "Inactif" pour qu’elles affichent "Actif" à la place.
// J’ai besoin de remplacer chaque "Temporaire" par "Permanent" dans le tableau.
// Crée une fonction qui change "Erreur" en "Corrigé" dans toutes les cellules de la feuille active.
// Dans mon Google Sheet, je veux que tous les "Ancien" soient remplacés par "Nouveau".
// Fais un script qui cherche "À faire" et remplace par "Terminé" dans toutes les lignes.
// Remplacer automatiquement "Client potentiel" par "Client confirmé" dans la feuille actuelle.
// Je souhaite qu’un script mette "En stock" à la place de "Rupture" partout dans le document.
function replaceTextInSheet() {
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
  const data = sheet.getDataRange();
  data.createTextFinder("oldText").replaceAllWith("newText");
}

// Ajoute une nouvelle ligne avec "Jean Dupont", "Marketing" et "Actif" dans ma feuille Google Sheets.
// Peux-tu m'insèrer une ligne à la fin avec les données : "Jean Dupont", "Marketing", "Actif" ?
// Je veux un script qui ajoute une ligne contenant "Jean Dupont", "Marketing" et "Actif" dans la feuille actuelle.
// Insère automatiquement une ligne en bas avec "Jean Dupont", "Marketing", "Actif".
// Remplis une nouvelle ligne avec les valeurs suivantes : nom = "Jean Dupont", service = "Marketing", statut = "Actif".
// Crée une fonction qui ajoute une entrée à la feuille : "Jean Dupont", "Marketing", "Actif".
// Ajoute une ligne à la fin du tableau avec ces informations : "Jean Dupont", "Marketing", "Actif".
// J’ai besoin d’un script pour écrire "Jean Dupont", "Marketing", "Actif" sur une nouvelle ligne.
// Programme un ajout de ligne avec ces champs : "Jean Dupont", "Marketing", "Actif".
// Je veux insérer une nouvelle ligne à la fin contenant "Jean Dupont", "Marketing", "Actif".
function addRowToSheet() {
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
  sheet.appendRow(["Jean Dupont", "Marketing", "Actif"]);
}

// Trie les données par nom en ordre alphabétique.
// Peux-tu trier la première colonne par ordre croissant ?
// Je veux organiser les lignes selon les noms (ordre A à Z).
// Fais un tri croissant sur la colonne des noms.
// Classe les données par ordre alphabétique sur la première colonne.
function sortByNameAsc() {
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
  const range = sheet.getDataRange();
  range.sort({ column: 1, ascending: true });
}

// Trie les données par nom en ordre alphabétique décroissant.
// Peux-tu trier la première colonne par ordre décroissant ?
// Je veux organiser les lignes selon les noms (ordre Z à A).
// Fais un tri décroissant sur la colonne des noms.
// Classe les données par ordre alphabétique (mais inversé) sur la première colonne.
// Tri par ordre décroissant les noms des clients.
function sortByNameDesc() {
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
  const range = sheet.getDataRange();
  range.sort({ column: 1, ascending: false });
}

// Trie les lignes par date de plus récente à plus ancienne.
// Classe les données de la colonne 3 en ordre décroissant.
// Je veux trier le tableau par date, la plus récente en haut.
// Peux-tu trier les lignes en fonction des dates (ordre inverse) ?
// Ordonne la feuille selon la colonne des dates, du plus récent au plus ancien.
// Peux-tu trier le tableau par performance, ordre décroissant ?
// Ordonne les données en fonction des scores de la colonne 5, du plus grand au plus petit.
function sortByDateDesc() {
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
  const range = sheet.getDataRange();
  range.sort({ column: 3, ascending: false });
}

// Trie la colonne du statut par ordre alphabétique.
// Organise les lignes selon les statuts.
// Je veux trier les données en fonction de la colonne "Statut".
// Classe les lignes de "Inactif" à "Actif".
// Effectue un tri croissant sur la 4e colonne.
function sortByStatusAsc() {
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
  const range = sheet.getDataRange();
  range.sort({ column: 4, ascending: true });
}

// Trie d’abord par département, puis par nom.
// Classe les données par service, et ensuite par ordre alphabétique du nom.
// Peux-tu organiser les lignes en fonction du département, puis du nom ?
// Je veux un tri croisé : d’abord la colonne 2, ensuite la 1.
// Ordonne le tableau par département (col. 2), puis par nom (col. 1).
function sortByDepartmentAndName() {
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
  const range = sheet.getDataRange();
  range.sort([
    { column: 2, ascending: true },  // Département
    { column: 1, ascending: true }   // Nom
  ]);
}

// Trie par score décroissant (du plus élevé au plus bas).
// Classe les lignes selon la colonne des notes, du plus haut au plus bas.
// Je veux que les meilleurs scores soient en haut du tableau.
// Peux-tu trier le tableau par performance, ordre décroissant ?
// Ordonne les données en fonction des scores de la colonne 5, du plus grand au plus petit.
// Effectue un tri décroissant sur la 5e colonne.
function sortByScoreDesc() {
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
  const range = sheet.getDataRange();
  range.sort({ column: 5, ascending: false });
}

// Crée une nouvelle feuille nommée "Transactions".
// Ajoute un onglet appelé "Transactions" dans mon Google Sheets.
// Ajoute un nouvel onglet "Transactions".
// Peux-tu créer une feuille vide avec le nom "Transactions" ?
// J’ai besoin d’un nouvel onglet intitulé "Transactions".
// Génère une nouvelle feuille nommée "Transactions" dans ce fichier.
function createNewSheet() {
  const spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  spreadsheet.insertSheet("Transactions");
}

// Renomme la feuille actuelle en "Clients".
// Change le nom de l’onglet actif en "Clients".
// Je veux que la feuille s’appelle maintenant "Clients".
// Mets "Clients" comme nom de la feuille actuelle.
// Modifie le nom de la feuille en "Clients".
function renameSheetToClients() {
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
  sheet.setName("Clients");
}

// Crée la feuille "Bilan 2025" si elle n’existe pas déjà.
// Je veux un onglet nommé "Bilan 2025", mais uniquement s’il n’est pas présent.
// Ajoute une feuille "Bilan 2025" uniquement si elle n'existe pas encore.
// Vérifie si la feuille "Bilan 2025" existe, sinon crée-la.
// Gère l’ajout de "Bilan 2025" uniquement si nécessaire.
function createSheetIfNotExists() {
  const spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  const sheetName = "Bilan 2025";
  if (!spreadsheet.getSheetByName(sheetName)) {
    spreadsheet.insertSheet(sheetName);
  }
}

// Supprime toutes les lignes dupliquées dans la feuille actuelle.
// Enlève les doublons dans mon tableau Google Sheets.
// Peux-tu nettoyer la feuille en supprimant les lignes en double ?
// Garde seulement les lignes uniques dans l’onglet actif.
// Je veux retirer toutes les doublons et conserver une seule occurrence.
// Crée un script pour effacer les doublons dans toutes les colonnes.
// Écris une fonction pour supprimer les lignes identiques dans la feuille.
// Élimine les entrées dupliquées dans mon tableau.
// Nettoie les doublons de la feuille actuelle sans toucher au formatage.
// Je veux une fonction qui détecte et supprime toutes les lignes en double.
// Supprime les doublons dans toutes les colonnes de mon tableau.
// Je veux nettoyer la feuille en gardant uniquement les lignes uniques.
// Détecte les lignes répétées et supprime-les automatiquement.
// Peux-tu me faire une fonction pour retirer les lignes en double dans ce fichier ?
// Retire tous les doublons de cette feuille Google Sheets.
function deleteDuplicateRows() {
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
  const data = sheet.getDataRange().getValues();
  const uniqueData = [];
  const seen = new Set();
  
  for (let i = 0; i < data.length; i++) {
    const row = data[i].join();
    if (!seen.has(row)) {
      seen.add(row);
      uniqueData.push(data[i]);
    }
  }
  
  sheet.clearContents();
  sheet.getRange(1, 1, uniqueData.length, uniqueData[0].length).setValues(uniqueData);
}

// Envoie un e-mail à l’adresse en colonne A, avec l’objet en B et le message en C.
// Je veux un script qui lit la ligne 2 et envoie un e-mail avec les infos.
// Peux-tu faire une fonction qui utilise les colonnes A, B et C pour envoyer un e-mail ?
// Utilise la première ligne de données pour envoyer un e-mail automatiquement.
// Crée une fonction qui envoie un message à l’adresse indiquée en A2.
// Génère un script pour envoyer un mail en prenant les infos de la ligne 2.
// Lis l’adresse email en A2, l’objet en B2, le contenu en C2, puis envoie un mail.
// J’aimerais envoyer un e-mail automatique depuis Google Sheets en utilisant les colonnes A, B, et C.
// Fait un script qui prend les infos de la deuxième ligne pour envoyer un email simple.
// Envoie un message automatiquement à partir des données saisies dans la ligne 2.
// Envoie un mail à partir des infos de la feuille : email, sujet, message.
// Automatise l’envoi d’un email simple en utilisant les colonnes A, B et C.
function sendEmailFromSheet() {
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
  const row = sheet.getRange(2, 1, 1, 3).getValues()[0]; // A2:C2
  const email = row[0];
  const subject = row[1];
  const message = row[2];

  MailApp.sendEmail(email, subject, message);
}

// Crée un événement dans mon agenda en utilisant la ligne 2 de la feuille.
// Ajoute un événement avec le titre en A2, le début en B2 et la fin en C2.
// Je veux planifier un événement Google Agenda à partir de données dans le tableau.
// Peux-tu générer un script qui lit le titre, la date de début et de fin dans la ligne 2 ?
// Lis les infos en A2, B2, et C2, puis ajoute-les à mon calendrier.
// Programme une fonction qui crée un événement Google Calendar depuis une feuille Google Sheets.
// Ajoute un événement intitulé selon la colonne A, débutant à la date en B, et se terminant à la date en C.
// Utilise la deuxième ligne de ma feuille pour ajouter un événement à Google Agenda.
// Je veux qu’un événement soit créé automatiquement depuis la ligne 2.
// Insère un événement dans mon agenda en lisant les colonnes A, B et C.
// Ajoute automatiquement un événement basé sur les infos dans la feuille (titre, date début, date fin).
// Gère l’ajout d’un événement avec comme titre la valeur de A2, début B2, fin C2.
function createCalendarEvent() {
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
  const row = sheet.getRange(2, 1, 1, 3).getValues()[0]; // A2:C2
  const title = row[0];
  const start = new Date(row[1]);
  const end = new Date(row[2]);

  const calendar = CalendarApp.getDefaultCalendar();
  calendar.createEvent(title, start, end);
}
