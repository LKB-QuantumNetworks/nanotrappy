// Open links externally.
var links = document.links;

for (var i = 0, linksLength = links.length; i < linksLength; i++) {
  if (links[i].hostname != window.location.hostname) {
    links[i].target = "_blank";
  }
}

$(document).ready(function () {
  $("#logoSVG").load("./assets/logos.svg", function () {
    $("#svg6290").click(function (evt) {
      switch (evt.target.id) {
        case "EU":
          window.open("https://europa.eu/", "_blank");
          break;
        case "DAALI":
          window.open("https://daali-project.eu/", "_blank");
          break;
        case "LKB":
          window.open("http://www.lkb.upmc.fr/", "_blank");
          break;
        case "SORBONNE":
          window.open("https://www.sorbonne-universite.fr/", "_blank");
          break;
        case "ANR":
          window.open("https://anr.fr/", "_blank");
          break;
        case "CNRS":
          window.open("https://www.cnrs.fr/", "_blank");
          break;
        case "ENS":
          window.open("https://www.ens.psl.eu/", "_blank");
          break;
        case "CDF":
          window.open("https://www.college-de-france.fr/", "_blank");
          break;
      }
    });
    $("#EU, #DAALI, #LKB, #SORBONNE, #CNRS, #ANR, #CDF, #ENS").hover(
      function () {
        $(this).css("cursor", "pointer");
      }
    );
  });
});
