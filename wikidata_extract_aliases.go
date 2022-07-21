package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sort"

	"github.com/hashicorp/go-retryablehttp"
	"gitlab.com/tozd/go/errors"

	"gitlab.com/tozd/go/mediawiki"
)

const (
	//wikidataTestDump = "https://gitlab.com/tozd/go/mediawiki/-/raw/main/testdata/wikidata-testdata-all.json.bz2"
	dumpFile = "wikidata-20220502-all.json.bz2"
	// dumpFile = "wikidata-testdata-all.json.bz2"
	language = "en"
)

func main() {
	ExtractaliasesAndTypes()
}

type EntityInformation struct {
	ForRelations []string `json:"r"`
	MainLabel    string   `json:"l"`
	Aliases      []string `json:"a"`
	ClaimCount   int      `json:"c"`
}

func ExtractaliasesAndTypes() {

	log.Println("Starting")

	types_of_interest := map[string][]string{
		"Q11344":   {"ChemicalCompoundElement"},                         // type of oxygen https://www.wikidata.org/wiki/Q629 and carbon https://www.wikidata.org/wiki/Q623
		"Q4830453": {"CompanyParentOrganization", "PersonEmployer"},     //business
		"Q6881511": {"CompanyParentOrganization", "PersonEmployer"},     //enterprise
		"Q786820":  {"CompanyParentOrganization", "PersonEmployer"},     //automobile manufacturer
		"Q891723":  {"CompanyParentOrganization", "PersonEmployer"},     //public company
		"Q134161":  {"CompanyParentOrganization", "PersonEmployer"},     //joint-stock company
		"Q783794":  {"CompanyParentOrganization", "PersonEmployer"},     //company
		"Q6256":    {"CountryBordersWithCountry", "RiverBasinsCountry"}, //country
		"Q3624078": {"CountryBordersWithCountry", "RiverBasinsCountry"}, // sovereign state
		"Q34770":   {"CountryOfficialLanguage", "PersonLanguage"},       //language
		"Q33742":   {"CountryOfficialLanguage", "PersonLanguage"},       //natural language
		//
		"Q1931388":   {"PersonCauseOfDeath"}, //cause of death
		"Q132821":    {"PersonCauseOfDeath"}, //crime against life
		"Q171558":    {"PersonCauseOfDeath"}, // accident
		"Q43229":     {"PersonEmployer"},     // organization
		"Q327333":    {"PersonEmployer"},     // government agency
		"Q17505024":  {"PersonEmployer"},     //space agency e.g., NASA
		"Q2659904":   {"PersonEmployer"},     // government organization
		"Q2385804":   {"PersonEmployer"},     // educational institution
		"Q3918":      {"PersonEmployer"},     // university
		"Q110295396": {"PersonInstrument"},   //type of musical instrument
		"Q7390":      {"PersonInstrument"},   //voice
		"Q29982117":  {"PersonInstrument"},   // musical instrument model
		"Q34379":     {"PersonInstrument"},   // musical instrument
		"Q1093829":   {"PersonPlaceOfDeath"}, // city of the United States
		"Q1549591":   {"PersonPlaceOfDeath"}, //big city
		"Q174844":    {"PersonPlaceOfDeath"}, // megacity
		"Q515":       {"PersonPlaceOfDeath"}, // city
		"Q486972":    {"PersonPlaceOfDeath"}, // human settlement
		"Q28640":     {"PersonProfession"},   // profession
		"Q66715801":  {"PersonProfession"},   //musical profession
		"Q12737077":  {"PersonProfession"},   // occupation
		"Q88789639":  {"PersonProfession"},   // artistic profession
		// StateSharesBorderState the schema used for this is rather starnge on wikidata, making this list very incomplete. MC started from some examples from the dev set and lloked them up. Then picked the types of these entities.
		// There is a general state type Q106458883, but that appears to be used rarely. Also, many of the regions in the test set would not be called states in the country in question, but rather canton, county, region, etc.
		"Q193512":    {"StateSharesBorderState"}, // region of Finland
		"Q35657":     {"StateSharesBorderState"}, // U.S. state
		"Q106458883": {"StateSharesBorderState"}, // state
		"Q9357527":   {"StateSharesBorderState"}, // territory of Canada
		"Q23058":     {"StateSharesBorderState"}, // canton of Switzerland
		"Q16110":     {"StateSharesBorderState"}, // region of Italy
		"Q261543":    {"StateSharesBorderState"}, // state of Austria
		"Q200547":    {"StateSharesBorderState"}, // county of Sweden
		//"Q712378":    {"DEBUG"},
		//"Q331769":    {"DEBUG"},
	}

	client := retryablehttp.NewClient()
	client.RequestLogHook = func(logger retryablehttp.Logger, req *http.Request, retry int) {
		req.Header.Set("User-Agent", "Qualifier extraction for recommender")
	}

	curDir, _ := os.Getwd()
	dumpPath := filepath.Join(curDir, dumpFile)
	output, err := os.Create(filepath.Join(curDir, "aliases.jsonl"))
	if err != nil {
		log.Panic(err)
	}
	defer output.Close()

	counter := 0

	err_round1 := mediawiki.ProcessWikidataDump(
		context.Background(),
		&mediawiki.ProcessDumpConfig{
			// URL:                    wikidataTestDump,
			Path:                   dumpPath,
			Client:                 client,
			ItemsProcessingThreads: 1,
		},
		func(_ context.Context, a mediawiki.Entity) errors.E {

			if counter%100000 == 0 {
				log.Println("Now at entity ", counter)
			}
			counter++

			possibly_relevant_types := make([]mediawiki.Statement, 0)
			possibly_relevant_types = append(possibly_relevant_types, a.Claims["P31"]...)
			possibly_relevant_types = append(possibly_relevant_types, a.Claims["P279"]...)

			interesting_for_relations := make(map[string]interface{})

			for _, statement := range possibly_relevant_types {

				if statement.MainSnak.SnakType == mediawiki.Value {
					if statement.MainSnak.DataValue == nil {
						log.Fatal("Found a main snak with type Value, while it does not have a value. This is an error in the dump.")
					}
					value := statement.MainSnak.DataValue.Value
					switch v := value.(type) {
					default:
						log.Printf("Unexpected type %T", value)
					case mediawiki.WikiBaseEntityIDValue:
						type_q_number := v.ID
						if relations, ok := types_of_interest[type_q_number]; ok {
							for _, relation := range relations {
								interesting_for_relations[relation] = nil
							}
						}
					}
				} else {
					log.Printf("Found a type statement without a value: %v", statement)
				}
			}

			if len(interesting_for_relations) == 0 {
				return nil
			}

			this_language_aliases := make([]string, 0, len(a.Aliases[language]))
			for _, alias := range a.Aliases[language] {
				this_language_aliases = append(this_language_aliases, alias.Value)
			}

			this_language_label := a.Labels[language].Value

			interesting_for_relations_list := make([]string, 0, len(interesting_for_relations))
			for relation := range interesting_for_relations {
				interesting_for_relations_list = append(interesting_for_relations_list, relation)
			}
			// sorting is not strictly necessary, but guarantees consistency between runs and probably helps a bit making it compressable.
			sort.Strings(interesting_for_relations_list)

			number_of_claims := len(a.Claims)

			info := EntityInformation{
				ForRelations: interesting_for_relations_list,
				MainLabel:    this_language_label,
				Aliases:      this_language_aliases,
				ClaimCount:   number_of_claims,
			}
			res, err := json.Marshal(info)
			if err != nil {
				log.Panic(err)
			}

			fmt.Fprintln(output, string(res))

			return nil
		},
	)

	if err_round1 != nil {
		log.Panic("An error occured during the first pass ", err_round1)
	}

}
