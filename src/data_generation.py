"""
data_generation.py
==================
Generates a synthetic moral dataset for the Moral Compass Classifier.

Classes:
  - Utilitarian  : Actions that maximize overall benefit / greater good
  - Ethical      : Morally principled, fair, honest behaviour (mapped from NTA)
  - Selfish      : Self-centred, harmful or dishonest behaviour (mapped from YTA)

Includes English + Indian/Hinglish scenarios for diversity.
"""

import pandas as pd
import os

# ---------------------------------------------------------------------------
# UTILITARIAN SCENARIOS (maximize overall good / sacrifice for the many)
# ---------------------------------------------------------------------------
UTILITARIAN_SCENARIOS = [
    # English
    "I donated my kidney to a stranger because it would save their life and I only need one.",
    "I voted for higher taxes on the wealthy so that public schools could be better funded for all children.",
    "As a doctor, I chose to allocate the last ventilator to the younger patient who had a better survival chance.",
    "I agreed to work overtime without extra pay because my absence would have caused the whole team to miss a critical deadline.",
    "I reported a corrupt official even though it put me at personal risk, because thousands of people were being harmed.",
    "I decided to vaccinate my child to contribute to herd immunity and protect vulnerable people in my community.",
    "I blew the whistle on my company's illegal pollution even though it cost me my job, because thousands lived near the river.",
    "I diverted a runaway trolley onto a side track where it would only hit one person instead of five.",
    "I chose a career in public health over a higher-paying private role because I could help more people.",
    "I sacrificed my weekend to volunteer at flood relief camps because the community needed more hands.",
    "I advocated for open-source software in my organization because it benefits the wider developer community.",
    "I chose to adopt an older child from an orphanage because younger children are adopted more easily.",
    "I supported a policy that slightly raises my taxes but provides free meals to thousands of hungry school kids.",
    "I gave up my business-class upgrade so a disabled veteran could travel comfortably.",
    "I pushed for remote work at my firm because it reduces the carbon footprint for the entire workforce.",
    "I lobbied for stricter food safety laws even though it made my business more expensive to operate.",
    "I agreed to a painful experimental treatment that, if successful, could cure a disease affecting millions.",
    "I decided to use my inheritance to fund clean water wells in rural villages rather than buy a luxury car.",
    "I supported euthanasia laws because they reduce prolonged suffering for terminally ill patients and their families.",
    "I chose to be an organ donor because after death my organs can give life to up to eight people.",
    "As a judge, I sentenced a beloved community leader to prison because letting him go would undermine the rule of law for everyone.",
    "I reported my neighbour's illegal dumping even though it ruined our friendship, because the toxins were poisoning the local water supply.",
    "I ate less meat this year to reduce my environmental footprint and help fight climate change globally.",
    "I chose to live in a smaller apartment so I could donate more money to effective charities.",
    "I supported mandatory helmet laws even though I personally find them inconvenient, because they save thousands of lives yearly.",
    "I signed up for a clinical trial for a new vaccine knowing there were risks, because it could protect millions.",
    "I voted to close my local polluting factory even though it meant job losses, because the air quality was harming children.",
    "I agreed to a price cap on my product that reduced my profits but made it accessible to low-income families.",
    "I advocated against single-use plastics in my restaurant at the cost of convenience, for the sake of the ocean.",
    "I chose to share my proprietary medical research findings publicly so that global scientists could develop a cure faster.",
    # Indian / Hinglish
    "Maine apni purani gaadi donate kar di gaon ke school ke liye, kyunki wahan bacchon ko paidal 10 km jaana padta tha.",
    "Mujhe pata tha meri complaint se meri naukri jaayegi, lekin maine factory ki pollution report ki kyunki poore mohalle ke log beemar pad rahe the.",
    "Maine apni behen ki shaadi cancel karwane mein help ki jab pata chala ladke ka background criminal tha — hamare ghar ko takleef hogi but uski zindagi bachegi.",
    "Maine vote diya sarkari school funding badhane ke liye kyunki private school mein sirf ameer log padhte hain.",
    "Maine apna bone marrow donate kiya ek anjaan bacche ko jo leukemia se jujh raha tha.",
    "Maine apne colony ke bhrashtachar ko expose kiya, chahe sab ne mujhse baat karna band kar diya.",
    "Maine garib students ke liye free tuition dena shuru kiya kyunki unke paas coaching ki afford karne ki aukaat nahi thi.",
    "Maine apni savings se ek village mein solar panels lagate waqt koi credit nahi liya — result matter karta hai, naam nahi.",
    "Maine apni compay ki secret pollution data leak ki NGO ko, chahe meri job jaane ka darr tha — thousands log clean water deserve karte hain.",
    "Mujhe mera promotion milne wala tha, par maine apni colleague ko recommend kiya kyunki uski family zyada zarooratmand thi.",
    "Maine ek badi company ka offer reject kiya aur government hospital mein kaam kiya kyunki wahan doctors ki bahut kami thi.",
    "Maine apni luxury car bechkar poor students ke liye scholarship fund banaya.",
    "Maine apne area mein tree plantation drive organize ki aur 500 ped lagaye kyunki temperature har saal badh raha hai.",
    "Maine apne ghar ke bahar free wifi lagaya taaki bachche online padh sakein.",
    "Maine blood donate kiya emergency mein kyunki kisi anjaan ki jaan khtre mein thi.",
    "Maine ek accident site par ruk kar injured stranger ki help ki, chahe mujhe interview ke liye late hona pada.",
    "Maine apni pocket money garib logo ko khana khilane mein kharch ki Ramzan ke mahine mein.",
    "Maine apne colony ke saare lights LED se replace karne ki petition sign ki aur organize ki, bijli bacha kar sab ka faayda hoga.",
    "Maine apna kidney donate kiya apne anjaan patient ko kyunki 5 saal se koi donor nahi mila tha.",
    "Maine apni medical research sab ke liye open-source kar di taaki dusre scientists bhi iske upar kaam kar sakein.",

    # More English
    "I pushed for universal basic income in my country because it would lift millions out of poverty.",
    "I chose to share my water supply with neighboring villages during a drought even though it reduced my own reserves.",
    "I agreed to a biodiversity zone on my farmland, reducing my harvest but protecting endangered species.",
    "I supported tougher drunk-driving penalties because they save thousands of lives even though it inconveniences many.",
    "I donated my prize money to a disaster relief fund instead of keeping it for myself.",
    "I advocated for free public transport to reduce urban congestion and carbon emissions, even though I own a car.",
    "I gave up my seat on a lifeboat so a young mother with a baby could survive.",
    "I organized a community kitchen during the pandemic so that daily-wage workers would not go hungry.",
    "I introduced a four-day work week at my company at the cost of some productivity, because employee mental health is vital.",
    "I chose to testify against a powerful politician even though my safety was at risk, because justice for the victims mattered more.",
]

# ---------------------------------------------------------------------------
# ETHICAL SCENARIOS (principled, fair, honest — NTA equivalent)
# ---------------------------------------------------------------------------
ETHICAL_SCENARIOS = [
    # English
    "I refused to lie to my boss about my colleague's mistake even though it meant extra scrutiny for the team.",
    "I told my friend the truth about their business plan even though it hurt their feelings.",
    "I returned the extra change the cashier gave me by mistake because it was the right thing to do.",
    "I declined a bribe from a contractor even though accepting it would have benefited me personally.",
    "I reported my friend's cheating on the exam because academic integrity matters.",
    "I did not take credit for my teammate's work during the presentation.",
    "I apologized to my neighbour after my dog damaged their garden, and I paid for the repair.",
    "I gave an honest performance review to my employee instead of inflating scores to avoid conflict.",
    "I refused to sign a misleading document my employer pressured me to sign.",
    "I returned a lost wallet to its owner with all the money intact.",
    "I told the car dealership that they had undercharged me, and I paid the correct amount.",
    "I did not shade the truth in my job application even though it could have gotten me the role.",
    "I stepped down from a committee when I realized I had a conflict of interest.",
    "I admitted my mistake in the report to my supervisor even though it was embarrassing.",
    "I did not take my sibling's side in a dispute just because they are family — I stood by what was right.",
    "I told my partner the truth about a past mistake that was bothering my conscience.",
    "I refused to join the office gossip about a colleague who wasn't present.",
    "I corrected a misunderstanding in court even when the truth was unfavorable to me.",
    "I shared the credit with my co-author equally, even though I did more of the work.",
    "I refused to misrepresent my product's capabilities to a potential customer.",
    "I told my friend their relationship was unhealthy, even though they got angry at me.",
    "I informed a home buyer about a defect in the house even though it caused the deal to fall through.",
    "I returned a borrowed book even years later because I had promised I would.",
    "I disclosed a medical error to the patient's family even though it exposed the hospital to a lawsuit.",
    "I did not cheat on my taxes even though I knew I would not get caught.",
    "I kept a secret my friend shared with me in confidence, even when others pressured me to reveal it.",
    "I stood up against bullying of a classmate even though the bully was popular.",
    "I gave truthful feedback on a friend's article even though they wanted only praise.",
    "I refused to manipulate a vulnerable elderly relative into changing their will.",
    "I declined an invitation to a party held at a venue that was known to discriminate against minorities.",
    # Indian / Hinglish
    "Maine apne dost ki cheating report ki exam mein, chahe usne mujhse baat karna band kar diya.",
    "Cashier ne galti se zyada change de diya, maine wapas kar diya kyunki yahi sahi hai.",
    "Maine apni galti apne boss ko bata di, chahe mujhe daant padne ka darr tha.",
    "Maine bribe lene se mana kar diya contractor se, chahe mujhe faayda hota.",
    "Mera padosi poochh raha tha kuch — maine usse sachchi salah di, chaahe mujhe faayda nahi tha.",
    "Maine apne dost ki business plan ki kamzoriyan bataai sachchi tarah, chahe unhe bura laga.",
    "Maine ek kho gayi purse wapas ki bina kuch liye, wallet mein paise the.",
    "Maine apne colleague ka kaam credit nahi liya presentation mein.",
    "Maine apni galti exam mein teacher ko bata di, seedhi baat sach ke saath.",
    "Maine jhooth bolne se mana kar diya apni company ke liye misleading document sign karne se.",
    "Maine apne friend ko sachchi feedback di unki kahani ke baare mein, chahe unhe dard hua.",
    "Maine admit kiya ki building mein ek defect hai ghar kharidne wale ko, deal cancel ho gayi par sach zaroori tha.",
    "Maine dono ko equal credit diya project mein, chahe main zyada kaam kiya.",
    "Maine gossip mein participate karne se mana kar diya colleague ke baare mein.",
    "Maine apna conflict of interest disclose kiya committee mein aur khud ko withdraw kar liya.",
    "Padosi ke baad meri gaadi se unka property damage hua, maine maafi mangi aur repair ke paise diye.",
    "Maine apne relative ko manipulate karne se mana kiya will badlwane ke liye.",
    "Maine apne saathi ko bully hote dekha aur uske khilaaf khada hua, chahe bully popular tha.",
    "Maine apne interview mein sach bola, false experience add nahi ki.",
    "Maine ek elderly patient ko medical error ke baare mein bataya, chahe hospital ko nuksan tha.",

    # More English
    "I did not falsely accuse my ex-partner to gain an advantage in the custody dispute.",
    "I let a competitor win a tender because I realized I had inadvertently seen their confidential proposal.",
    "I gave the job to the more qualified candidate even though the other was my friend.",
    "I informed the inspector of the genuine safety issues at my factory rather than bribing them to overlook it.",
    "I returned excess insurance payout to the company because I knew I was overcompensated.",
    "I refused to help my child cheat on a school assignment even though they were stressed.",
    "I corrected my professor when they cited wrong data in class, politely but clearly.",
    "I told my employer I needed a day off for personal reasons instead of pretending to be sick.",
    "I fulfilled a verbal promise to an old friend even though I had nothing in writing.",
    "I did not shade the truth in my medical history form even though it might affect my insurance.",
]

# ---------------------------------------------------------------------------
# SELFISH SCENARIOS (self-centred, harmful, dishonest — YTA equivalent)
# ---------------------------------------------------------------------------
SELFISH_SCENARIOS = [
    # English
    "I took credit for my intern's work during the annual review and got a promotion.",
    "I parked in the disabled parking spot because I was in a hurry and it was just for five minutes.",
    "I cheated on my partner for months but never admitted it because it was convenient for me.",
    "I lied on my resume about my qualifications to get a job I wasn't qualified for.",
    "I ignored a homeless person asking for food and walked past them.",
    "I manipulated my elderly parent into changing their will in my favor.",
    "I cut in line at the hospital claiming it was urgent just because I didn't want to wait.",
    "I used my friend's credit card without telling them because I forgot my wallet.",
    "I got my neighbor's package delivered to my door and kept it.",
    "I cheated in my exam because I hadn't studied and needed to pass.",
    "I stole a parking spot someone was clearly waiting for.",
    "I ghosted a friend who needed emotional support because I found it inconvenient.",
    "I took the last elevator spot and let the pregnant woman wait for the next one.",
    "I lied to my employee to avoid paying them a raise they deserved.",
    "I broke up with my partner over text on their birthday because I was too cowardly to face them.",
    "I spread a rumour about my colleague to make myself look better at work.",
    "I borrowed money from a struggling friend and never paid it back.",
    "I blew the whistle on my coworker to remove competition for a promotion.",
    "I took food meant for a charity event because no one was watching.",
    "I pretended to be sick to avoid attending a friend's important event.",
    "I cut corners on the construction project to save money, knowing it created safety risks.",
    "I used my child's college fund for a luxury vacation I wanted.",
    "I took a wallet I found on the road and stole all the cash inside it.",
    "I found a dropped purse and kept the money instead of returning it.",
    "I saw someone drop a $100 bill and quietly put it in my pocket.",
    "I reported a competitor to regulators using false information to eliminate them from the market.",
    "I fired an employee just before their stock options vested to save the company money.",
    "I refused to wear a mask during a respiratory illness because it was uncomfortable for me.",
    "I charged a vulnerable elderly client for services I never actually performed.",
    "I took a bribe to approve a faulty building project because the money was too good to refuse.",
    "I ghosted a business partner after they had done all the work, to avoid paying their share.",
    "I deliberately gave a negative reference for a former colleague I was jealous of.",
    "I used company expenses for personal trips and claimed them as business travel.",
    # Indian / Hinglish
    "Maine apne intern ka kaam chura liya aur boss ke saamne apna bataya — promotion mil gayi.",
    "Maine disabled parking mein gaadi khari ki kyunki mujhe jaldi thi — 'bas 5 minute ki baat hai'.",
    "Maine apne partner se 6 mahine tak cheating ki aur kabhi admit nahi kiya kyunki convenient tha.",
    "Maine apni resume mein jhooth bola degree ke baare mein job paane ke liye.",
    "Mujhe pata tha mera padosi loan mein doob raha hai, par maine usse paise dene se mana kar diya aur apne ghumne chala gaya.",
    "Maine apne budhape mein baap ko manipulate kiya will mein apna naam daalne ke liye.",
    "Maine line mein cut kiya hospital mein yeh bolkar ki emergency hai — sirf wait nahi karna tha.",
    "Maine apne dost ka credit card bina bole use kiya.",
    "Maine galti se deliver hua package — neighbour ka tha — rakh liya.",
    "Maine exam mein cheating ki kyunki padha nahi tha aur paas hona zaroori tha.",
    "Maine apne colleague ke baare mein jhoothi baar baat failaai apni image improve karne ke liye.",
    "Maine apne dost se paise liye aur vapas dene ki niyat nahi thi.",
    "Maine charity ke liye rakha hua khana khaa liya kyunki koi dekh nahi raha tha.",
    "Maine apne employee ko jhooth bola raise nahi dene ke liye.",
    "Maine apne competitor ke baare mein false complaint file ki taaki unhe hataya ja sake.",
    "Maine bribe liya aur ek faulty building project approve kar diya paise ke lalach mein.",
    "Maine apni company ke kharche pe personal trips ki aur business mein dikhaaya.",
    "Maine apni baat manaane ke liye apni maa ko emotionally blackmail kiya.",
    "Maine apni ex ko false case mein fansane ki koshish ki custody fight mein apni jagah strong karne ke liye.",
    "Maine apne dost ki izzat ek party mein tod di sirf hasne ke liye.",
    "Sadak pe gira hua batua uthaya aur usme se saare paise nikal liye, bacha hua phek diya.",
    "Maine purse mila kisi aur ka aur usme se cash chura liya.",

    # More English
    "I gave a five-star review to my own restaurant using fake accounts to improve rankings.",
    "I let my colleague take the blame for a mistake that was mine.",
    "I returned a used product to the store claiming it was defective to get a refund.",
    "I told my landlord I was unemployed to get a lower rent, even though I had a good salary.",
    "I used my sick leave to go on vacation and then lied to my manager about it.",
    "I sold counterfeit goods to customers, knowing they were fake.",
    "I hid information from my business partner to force them to sell their share cheaply.",
    "I borrowed my sibling's savings and invested it in a scheme without telling them, then lost the money.",
    "I deliberately withheld praise from a teammate to keep them from getting recognized.",
    "I downloaded pirated software and refused to pay for tools my work required.",
]

# ---------------------------------------------------------------------------
# BUILD DATAFRAME
# ---------------------------------------------------------------------------
def generate_dataset() -> pd.DataFrame:
    records = []
    for text in UTILITARIAN_SCENARIOS:
        records.append({"text": text.strip(), "label": "Utilitarian"})
    for text in ETHICAL_SCENARIOS:
        records.append({"text": text.strip(), "label": "Ethical"})
    for text in SELFISH_SCENARIOS:
        records.append({"text": text.strip(), "label": "Selfish"})
    return pd.DataFrame(records)


if __name__ == "__main__":
    df = generate_dataset()
    print(f"Dataset size: {len(df)} rows")
    print(df["label"].value_counts())

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "synthetic_moral_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
