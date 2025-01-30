import os

import pandas as pd
"""
Excluded names are names that can have different meanings than just a name.

Determined mostly by checking meaning fields of the Google Translator

Examples:
- Mark: to mark something
- Henry: SI unit of inductance
- Carol: a religious folk song, particularly associtated with Christmas
- Christian: a believer in Christianity
- Mike: a microphone
"""
# todo move it to a file
excluded_names = {
    'John', # a prostitutes client (informal) and synonym to loo/can/bog
    'Mark', # a small area on a surface having a different color from its surroundings, typically one caused by accident or damage.
    'Timothy', # a Eurasian grass which is widely grown for grazing and hay. It is naturalized in North America, where many cultivars have been developed.
    'Karen', # a member of an indigenous people of eastern Burma (Myanmar) and western Thailand.
    'Frank', # open, honest, and direct in speech or writing, especially when dealing with unpalatable matters.
    'Donna', # a title or form of address for an Italian woman.
    'Ruth', # a feeling of pity, distress, or grief.
    'Carol', # a religious folk song or popular hymn, particularly one associated with Christmas.
    'Jack', # a device for lifting heavy objects, especially one for raising the axle of a motor vehicle off the ground so that a wheel can be changed or the underside inspected.
    'Henry', # the SI unit of inductance, equal to an electromotive force of one volt in a closed circuit with a uniform rate of change of current of one ampere per second.
    'Virginia', # a type of tobacco grown and manufactured in Virginia.
    'Jerry', # (rare) synonym to pot/jackpot and throne
    'Douglas', # company
    'Heather', # a purple-flowered Eurasian heath that grows abundantly on moorland and heathland.
    'Christina', # a light four-wheeled horse-drawn carriage with a collapsible hood, seats for two passengers, and an elevated driver's seat in front.
    'Joyce', # a feeling of great pleasure and happiness
    'Grace', # simple elegance or refinement of movement.
    'Rose', # a prickly bush or shrub that typically bears red, pink, yellow, or white fragrant flowers, native to north temperate regions.
    'Julia', # an orange and black American butterfly with long, narrow forewings, found chiefly in tropical regions.
    'Jean', # heavy twilled cotton cloth, especially denim.
    'Willie', # informal synonym to several words (plonker, lout/boor/oaf, brush/paintbrush)
    'Roger', # your message has been received and understood (used in radio communication).
    'Terry', # a fabric with raised uncut loops of thread covering both surfaces, used especially for towels.
    'Christian', # believer in Christianity
    'Harry', # persistently carry out attacks on (an enemy or an enemy's territory).
    'Austin', # city
    'Gloria', # a Christian liturgical hymn or formula beginning (in the Latin text) with Gloria.
    'Amber', # hard translucent fossilized resin produced by extinct coniferous trees of the Tertiary period, typically yellowish in color.
    'Jordan', # country
    'Brittany', # region in France
    'Billy', # short for billy goat and synonym to cookware
    'Diana', # a North American fritillary (butterfly), the male of which is orange and black and the female blue and black.
    'Charlotte', # a dessert made of stewed fruit or mousse with a casing or covering of bread, sponge cake, ladyfingers, or breadcrumbs.
    'Crystal', # a piece of a homogeneous solid substance having a natural geometrically regular form with symmetrically arranged plane faces.
    'Ruby', # a precious stone consisting of corundum in color varieties varying from deep crimson or purple to pale rose.
    'Erin', # archaic or poetic/literary name for Ireland.
    'Florence', # city
    'Randy', # sexually aroused or excited.
    'Bradley', # a medium-sized tank equipped for use in combat.
    'Victor', # a person who defeats an enemy or opponent in a battle, game, or other competition.
    'Martin', # a swift-flying insectivorous songbird of the swallow family, typically having a less strongly forked tail than a swallow.
    'Bobby', # synonym of policeman
    'Clarence', # a closed horse-drawn carriage with four wheels, seating four inside and two outside next to the coachman.
    'Ernest', # rare synonym to serieous
    'Dawn', # the first appearance of light in the sky before sunrise.
    'Robin', # a large New World thrush that typically has a reddish breast
    'Peggy', # a steward in a ship's mess (often used as a form of address).
    'Earl', # a British nobleman ranking above a viscount and below a marquess.
    'Jimmy', # a short crowbar used by a burglar to force open a window or door.
    'Mason', # a builder and worker in stone.
    'Dale', # a valley, especially a broad one.
    'Norma', # company and synonym of angle measurement
    'Hazel', # a temperate shrub or small tree with broad leaves, bearing prominent male catkins in spring and round hard-shelled edible nuts in autumn
    'Norman', # a member of a people of mixed Frankish and Scandinavian origin who settled in Normandy from about AD 912 and became a dominant military power in western Europe and the Mediterranean in the 11th century.
    'Jasmine', # an Old World shrub or climbing plant that bears fragrant flowers used in perfumery or tea. It is popular as an ornamental.
    'Chad', # a piece of waste material removed from card or tape by punching.
    'April', # month
    'Erica', # a plant of the genus Erica (family Ericaceae ), especially (in gardening) heather.
    'Sheila', # synonym of doll/puppet/dolly
    'Hunter', # a person or animal that hunts.
    'Lee', # the sheltered side of something; the side away from the wind.
    'Carolin', # relating to the reigns of Charles I and II of England.
    'Sherry', # a fortified wine originally and mainly from southern Spain, often drunk as an aperitif.
    'Angel', # a spiritual being believed to act as an attendant, agent, or messenger of God, conventionally represented in human form with wings and a long robe.
    'Jesus', # Jesus Christ
    'Pauline', # relating to or characteristic of St. Paul, his writings, or his doctrines.
    'Veronica', # a herbaceous plant of north temperate regions, typically with upright stems bearing narrow pointed leaves and spikes of blue or purple flowers.
    'Morgan', # a horse of a light thickset breed developed in New England.
    'Troy', # a system of weights used mainly for precious metals and gems, with a pound of 12 ounces or 5,760 grains.
    'Julian', # of or associated with Julius Caesar (adjective).
    'Holly', # a widely distributed shrub, typically having prickly dark green leaves, small white flowers, and red berries.
    'Lorraine', # region of France
    'Leo', # synonym of lion
    'Sally', # a sudden charge out of a besieged place against the enemy; a sortie.
    'Bertha', # a deep collar, typically made of lace, attached to the top of a dress that has a low neckline.
    'Oscar', # a South American cichlid fish with velvety brown young and multicolored adults, popular in aquariums.
    'Calvin', # company (Calvin Klein)
    'Mike', # a microphone.
    'Ray', # each of the lines in which light (and heat) may seem to stream from the sun or any luminous body, or pass through a small opening.
    'June', # month
    'Jay', # a bird of the crow family with boldly patterned plumage, typically having blue feathers in the wings or tail.
    'Barry', # divided into a number of equal horizontal bars of alternating tinctures.
    'Dean', # head of college
    'Warren', # a network of interconnecting rabbit burrows.
    'Regina', # (in the UK) the reigning queen (used following a name or in the titles of lawsuits, e.g., Regina v. Jones, the Crown versus Jones).
    'Sydney', # city
    'Chelsea', # Chelsea Football Club
    'Savannah', # a large, flat area of land covered with grass, usually with few trees, that is found in hot countries, especially in Africa
    'Charlie', # informal cocaine.
    'Molly', # a small, livebearing freshwater fish
    'Chase', # pursue in order to catch or catch up with
    'Don', # a Spanish title prefixed to a male forename
    'Carter', # synonym to coachman
    'Beth', # the second letter of the Hebrew alphabet.
    'Rosemary', # an evergreen aromatic shrub of the mint family
    'Bill', # an amount of money owed for goods
    'Georgia', # country
    'Pearl', # a hard, lustrous spherical mass, typically white or bluish-gray, formed within the shell of a pearl oyster or other bivalve mollusk and highly prized as a gem.
    'Max', # maximum
    'Lily', # a bulbous plant with large trumpet-shaped, typically fragrant, flowers on a tall, slender stem. Lilies have long been cultivated, some kinds being of symbolic importance and some used in perfumery.
    'Lewis', # a steel device for gripping heavy blocks of stone or concrete for lifting, consisting of three pieces arranged to form a dovetail, the outside pieces being fixed in a dovetail mortise by the insertion of the middle piece.
    'Derrick', # a kind of crane with a movable pivoted arm for moving or lifting heavy weights, especially on a ship.
    'Destiny', # the events that will necessarily happen to a particular person or thing in the future.
    'Marc', # the refuse of grapes or other fruit that has been pressed for winemaking
    'Violet', # color
    'Sue', # institute legal proceedings against (a person or institution), typically for redress.
    'Cole', # a brassica, especially cabbage, kale, or rape
    'Daisy', # a small grassland plant that has flowers with a yellow disk and white rays. It has given rise to many ornamental garden varieties.
    'Carmen', # a driver of a streetcar or horse-drawn carriage.
    'Constance', # noun of constant
    'Tom', # the male of various animals, especially a turkey or domestic cat.
    'Joy', # a feeling of great pleasure and happiness.
    'Faith', # complete trust or confidence in someone or something.
    'Franklin', # a landowner of free but not noble birth in the 14th and 15th centuries in England.
    'Ivan', # a Russian man, especially a Russian soldier
    'Francisco', # San Francisco
    'Sofia', # city
    'Garret', # a top-floor or attic room, especially a small dismal one (traditionally inhabited by an artist).
    'Myrtle', # an evergreen shrub which has glossy aromatic foliage and white flowers followed by purple-black oval berries.
    'Cora', # a member of an indigenous people of western Mexico.
    'Herman', # "her husband"
    'Grant', # agree to give or allow (something requested) to
    'Glen', # a narrow valley
    'Viola', # an instrument of the violin family, larger than the violin and tuned a fifth lower.
    'Autumn', # season
    'Gene', # a unit of heredity which is transferred from a parent to offspring and is held to determine some characteristic of the offspring.
    'Spencer', # a short, close-fitting jacket, worn by women and children in the early 19th century.
    'Patsy', # a person who is easily taken advantage of, especially by being cheated or blamed for something.
    'Miranda', # denoting or relating to the duty of the police to inform a person taken into custody of their right to legal counsel and the right to remain silent under questioning
    'Brandy', # a strong alcoholic spirit distilled from wine or fermented fruit juice.
    'Lance', # a long weapon for thrusting, having a wooden shaft and a pointed steel head, formerly used by a horseman in charging.
    'Dan', # any of ten degrees of advanced proficiency in judo or karate.
    'Brooklyn', # region of NYC
    'Bethany', # Palestinian town
    'Hector', # talk to (someone) in a bullying way.
    'Alexandria', # city in Egypt
    'Sierra', # a long jagged mountain chain
    'Bayley', # the outer wall of a castle
    'Penny', # used for emphasis to denote no money at all.
    'Maya', # a member of an indigenous people of Yucatán and adjacent areas.
    'Melody', # a sequence of single notes that is musically satisfying.
    'Angle', # the space (usually measured in degrees) between two intersecting lines or surfaces at or close to the point where they meet.
    'Kay', # short for ok
    'Maxwell', # a unit of magnetic flux in the centimeter-gram-second system, equal to that induced through one square centimeter by a perpendicular magnetic field of one gauss.
    'Jenny', # a female donkey or ass
    'Ada', # a high-level computer programming language used chiefly in real-time computerized control systems, e.g. for aircraft navigation.
    'Tanner', # a person who tans animal hides, especially to earn a living.
    'Marguerite', # another term for oxeye daisy
    'Angelica', # a tall aromatic plant of the parsley family, with large leaves and yellowish-green flowers. Native to both Eurasia and North America, it is used in cooking and herbal medicine.
    'Marshall', # a military officer of the highest rank
    'Bob', # a movement up and down
    'Guy', # (informal) a man.
    'Hope', # a feeling of expectation and desire for a certain thing to happen.
    'Jade', # a hard, typically green stone used for ornaments and implements and consisting of the minerals jadeite or nephrite.
    'Misty', # full of, covered with, or accompanied by mist
    'Cooper', # a maker or repairer of casks and barrels
    'Geneva', # city
    'Desiree', # a potato of a pink-skinned variety with yellow waxy flesh.
    'Dakota', # a member of a North American people of the upper Mississippi valley and the surrounding plains.
    'Brad', # a small wire nail with a small, often asymmetrical head.
    'Miles', # a unit of linear measure equal to 1,760 yards (approximately 1.609 kilometres).
    'Iris', # a flat, colored, ring-shaped membrane behind the cornea of the eye, with an adjustable circular opening (pupil) in the center.
    'Rick', # a stack of hay, corn, straw, or similar material, especially one formerly built into a regular shape and thatched
    'Harper', # a musician, especially a folk musician, who plays a harp.
    'Summer', # season
    'Dalton', # a unit used in expressing the molecular weight of proteins, equivalent to atomic mass unit.
    'Drew', # simple past of 'to draw'
    'Nelson', # a wrestling hold in which one arm is passed under the opponent's arm from behind and the hand is applied to the neck ( half nelson ), or both arms and hands are applied ( full nelson ).
    'Trinity', # the Christian Godhead as one God in three persons: Father, Son, and Holy Spirit.
    'Perry', # an alcoholic drink made from the fermented juice of pears.
    'Devon', # an animal of a breed of red beef cattle.
    'Ted', # turn over and spread out (grass, hay, or straw) to dry or for bedding.
    'Stuart', # elating to the royal family ruling Scotland 1371–1714 and Britain 1603–49 and 1660–1714.
    'Cesar', # Roman general and statesman
    'Wade', # walk with effort through water or another liquid or viscous substance.
    'Kennedy', # J.F. Kennedy
    'Ariel', # a gazelle found in the Middle East and North Africa.
    'Roman', # relating to ancient Rome or its empire or people.
    'Kirk', # synonym of church
    'Sandy', # covered in or consisting mostly of sand
    'Opal', # a gemstone consisting of a form of hydrated silica, typically semitransparent and showing many small points of shifting color against a pale or dark ground.
    'Cheyenne', # a member of an Algonquian people formerly living between the Missouri and Arkansas Rivers but now on reservations in Montana and Oklahoma.
    'Flora', # the plants of a particular region, habitat, or geological period.
    'Genesis', # the origin or mode of formation of something.
    'Luther', # Martin Luther
    'Guadalupe', # island
    'Patty', # a small flat cake of minced or finely chopped food, especially meat.
    'Santiago', # capital of Chile
    'Israel', # country
    'Homer', # a home run
    'Celeste', # color (sky blue)
    'Mckenzie', # company McKenzie
    'Aurora', # a natural electrical phenomenon characterized by the appearance of streamers of reddish or greenish light in the sky
    'Rex', # the reigning king (used following a name or in the titles of lawsuits, e.g. Rex v. Jones : the Crown versus Jones).
    'Gage', # a valued object deposited as a guarantee of good faith.
    'Earnest', # resulting from or showing sincere and intense conviction.
    'Serenity', # the state of being calm, peaceful, and untroubled.
    'Olive', # a small oval fruit with a hard pit and bitter flesh, green when unripe and brownish black when ripe, used as food and as a source of oil.
    'Lacey', # synonym to sharpen
    'Alfredo', # a sauce for pasta incorporating butter, cream, garlic, and Parmesan cheese.
    'Joey', # a young kangaroo or other marsupial.
    'Kerry', # an animal of a breed of small black dairy cattle.
    'Ira', # abbreviation IRA
    'Aria', # a long accompanied song for a solo voice, typically one in an opera or oratorio
    'Ivy', # a woody evergreen Eurasian climbing plant, typically having shiny, dark green five-pointed leaves.
    'Kelvin', # the SI base unit of thermodynamic temperature
    'Easton', # several cities
    'Nick', # a small cut or notch
    'May', # month
    'Chance', # a possibility of something happening.
    'Allie', # synonym of ally
    'Noel', # Christmas, especially as a refrain in carols and on Christmas cards.
    'Christie', # a sudden turn in which the skis are kept parallel, used for changing direction fast or stopping short.
    'Salvador', # country El Salvador
    'Laurel', # any of a number of shrubs and other plants with dark green glossy leaves.
    'Ty', # informal abbreviation for 'Thank you'
    'Eula', # EULA: a contract between a software producer and the eventual user of the product, specifying the terms and conditions of use.
    'Piper', # a bagpipe player
    'Ginger', # a hot, fragrant spice made from the rhizome of a plant, which may be chopped or powdered for cooking, preserved in syrup, or candied.
    'Mercedes', # company (car manufacturer)
    'Ezekiel', # Book of Ezekiel in the Bible
    'Morgan', # a horse of a light thickset breed developed in New England.
    'August', # month
    'Sawyer', # a person who saws timber for a living.
    'Will', # future tense.
    'Jasper', # an opaque reddish-brown variety of chalcedony.
    'Camden', # city
    'Alexia', # the inability to see words or to read, caused by a defect of the brain.
    'Dixie', # synonym to forecastle
    'Sylvester', # new year's eve
    'Madeleine', # a small rich cake, typically baked in a shell-shaped mold and often decorated with coconut and jam.
    'Ebony', # heavy blackish or very dark brown timber from a mainly tropical tree.
    'Axel', # a jump in skating with a forward takeoff from the forward outside edge of one skate to the backward outside edge of the other, with one and a half turns in the air.
    'Tucker', # a piece of lace or linen worn in or around the top of a bodice or as an insert at the front of a low-cut dress.
    'Griffin', # a mythical creature with the head and wings of an eagle and the body of a lion, typically depicted with pointed ears and with the eagle's legs taking the place of the forelegs.
    'Ollie', # (in skateboarding and snowboarding) a jump performed without the aid of a takeoff ramp, executed by pushing the back foot down on the tail of the board, bringing the board off the ground.
    'Junior', # a person who is a specified number of years younger than someone else.
    'Raven', # a large heavily built crow with mainly black plumage, feeding chiefly on carrion.
    'Jewel', # a precious stone, typically a single crystal or piece of a hard lustrous or translucent mineral cut into shape with flat facets or smoothed and polished for use as an ornament.
    'Fabian', # a member or supporter of the Fabian Society, an organization of socialists aiming at the gradual rather than revolutionary achievement of socialism
    'Clarissa', # a cocktail made with tequila and citrus fruit juice.
    'Beau', # a person's boyfriend or male admirer.
    'Pat', # touch quickly and gently with the flat of the hand.
    'Moses', # prophet in the bible
    'Aliyah', # immigration to Israel.
    'Marina', # a specially designed harbor with moorings for pleasure craft and small boats.
    'Bonita', # synonym to nice/beautiful
    'Lane', # a narrow road, especially in a rural area.
    'Paisley', # a distinctive intricate pattern of curved feather-shaped figures based on an Indian pine-cone design.
    'Clay', # a stiff, sticky fine-grained earth, typically yellow, red, or bluish-gray in color and often forming an impermeable layer in the soil. It can be molded when wet, and is dried and baked to make bricks, pottery, and ceramics.
    'Daphne', # a small Eurasian shrub with sweet-scented flowers and, typically, evergreen leaves.
    'Roosevelt', # Franklin D. Roosevelt
    'Iva', # short for individual voluntary arrangement.
    'Fern', # a flowerless plant which has feathery or leafy fronds and reproduces by spores released from the undersides of the fronds. Ferns have a vascular system for the transport of water and nutrients.
}



def apply_split(df, split):
    if split is None:
        return df
    elif split == 'train':
        return df[df['split'] == 'train'].reset_index(drop=True)
    elif split == 'val':
        return df[df['split'] == 'val'].reset_index(drop=True)
    elif split == 'test':
        return df[df['split'] == 'test'].reset_index(drop=True)
    else:
        raise ValueError(f"Invalid split: {split}")


def read_names_data(*, split=None, filter_non_unique=True, minimum_count=20000, gender_agreement_threshold=None,
                    force=False, filter_excluded_words=True, max_entries=None, subset=None):
    raw_file_name = 'name_gender_dataset.csv'
    cached_file = f"data/cache/name_gender_dataset/{raw_file_name.removesuffix('.csv')}_{filter_non_unique}_{minimum_count}_{gender_agreement_threshold}_{filter_excluded_words}_{max_entries}.csv"
    file = f'data/{raw_file_name}'

    if not force:
        try:
            df = pd.read_csv(cached_file)
            return apply_split(df, split)
        except FileNotFoundError:
            print('No cached names file found, generate new cache file...')

    df = pd.read_csv(file)
    df = df.rename(columns=str.lower)

    # filter out rare names
    if minimum_count:
        df = df[df['count'] >= minimum_count]

    # the dataset contains some names as both male and female names
    # we introduce an additional column reflecting this property as 'genders'
    def genders_agreement(row):
        rows = df[df['name'] == row['name']]

        if len(rows) > 1:
            rows = rows.set_index('gender')['count'].to_dict()
            count_M = rows['M']
            count_F = rows['F']
            primary_gender = 'M' if count_F < count_M else 'F'
            count = count_M + count_F
            prob_M = count_M / count
            prob_F = count_F / count
            agreement_ratio = max(prob_M, prob_F)
        else:
            primary_gender = rows['gender'].iloc[0]
            agreement_ratio = 1.0
            if primary_gender == 'M':
                prob_M = 1.0
                prob_F = 0.0
            else:
                prob_F = 1.0
                prob_M = 0.0

        return pd.Series([agreement_ratio, prob_M, prob_F, primary_gender],
                         index=['gender_agreement', 'prob_M', 'prob_F', 'primary_gender'])

    print('Calculating the genders agreement...')
    df[['gender_agreement', 'prob_M', 'prob_F', 'primary_gender']] = df.apply(genders_agreement, axis=1,
                                                                              result_type='expand')

    # filter out those names with both genders, where one gender dominates, so we filter out the less relevant gender row
    if gender_agreement_threshold:
        print(f'Filtering out gender agreements with less than {gender_agreement_threshold} probability')
        mask = (df['gender_agreement'] < gender_agreement_threshold) & (df['gender'] != df['primary_gender'])
        df.loc[mask, 'genders'] = df.loc[mask, 'gender']
        df = df.drop(df[mask].index).reset_index(drop=True)

    gender_counts = df.groupby('name')['gender'].transform('nunique')
    df['genders'] = df.apply(lambda row: 'B' if gender_counts.loc[row.name] > 1 else row.gender, axis=1)

    if filter_non_unique:
        df = df[df['genders'] != 'B']

    if filter_excluded_words:
        actual_excluded_names = [x for x in excluded_names if x in df['name'].unique()]
        print(f'Removing {len(actual_excluded_names)} ambiguous names')

        df = df[~df['name'].isin(excluded_names)].reset_index(drop=True)

    if max_entries and len(df) > max_entries:
        # sort by count and take the most frequent names
        df = df.sort_values('count', ascending=False).reset_index(drop=True)
        df = df.loc[range(max_entries)]

    if subset is not None:
        subset = subset()
        df = df[~df['name'].isin(subset['name'])].reset_index(drop=True)

    df_M = df[df['genders'] == 'M'].copy().sample(frac=1, random_state=42).reset_index(drop=True)
    df_F = df[df['genders'] == 'F'].copy().sample(frac=1, random_state=42).reset_index(drop=True)
    df_B = df[df['genders'] == 'B'].copy()

    test_prop = 0.1
    val_prop = 0.05

    n = len(df)
    test_n_per_group = int(n * test_prop / 2)
    val_n_per_group = int(n * val_prop / 2)

    if not test_n_per_group + val_n_per_group < min(len(df_F), len(df_M)):
        raise ValueError('The test and validation set sizes are too large!')

    df_M_test = df_M[:test_n_per_group]
    df_M_val = df_M[test_n_per_group:test_n_per_group + val_n_per_group]
    df_M_train = df_M[test_n_per_group + val_n_per_group:]

    df_F_test = df_F[:test_n_per_group]
    df_F_val = df_F[test_n_per_group:test_n_per_group + val_n_per_group]
    df_F_train = df_F[test_n_per_group + val_n_per_group:]

    df_train = pd.concat([df_M_train, df_F_train]).reset_index(drop=True)
    df_val = pd.concat([df_M_val, df_F_val]).reset_index(drop=True)
    df_test = pd.concat([df_M_test, df_F_test]).reset_index(drop=True)

    df_train['split'] = 'train'
    df_val['split'] = 'val'
    df_test['split'] = 'test'
    df_B['split'] = 'test'

    df = pd.concat([df_train, df_val, df_test, df_B]).reset_index(drop=True)

    if subset is not None:
        df = pd.concat([df, subset]).reset_index(drop=True)

    os.makedirs(os.path.dirname(cached_file), exist_ok=True)
    df.to_csv(cached_file, index=False)

    return apply_split(df, split)
