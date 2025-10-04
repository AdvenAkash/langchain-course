from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate 
from langchain_ollama import OllamaLLM
load_dotenv()

def main():
    print("Hello from langchain-course!")
    information = """
    Max Emilian Verstappen born 30 September 1997 is a Dutch and Belgian racing driver who competes under the Dutch flag in Formula One for Red Bull Racing. Verstappen has won four Formula One World Drivers' Championship titles, which he won consecutively from 2021 to 2024 with Red Bull, and has won 67 Grands Prix across 11 seasons.

    Born in Hasselt and raised in Maaseik, Verstappen is the son of Dutch former Formula One driver Jos Verstappen and Belgian former kart racer Sophie Kumpen. After a successful karting career—culminating in his record-breaking 2013 season—Verstappen graduated to junior formulae. Progressing directly to FIA European Formula 3, Verstappen broke several records on his way to third in the championship in his rookie season with Van Amersfoort.[b] Aged 17, Verstappen signed for Toro Rosso in 2015 as part of the Red Bull Junior Team, becoming the youngest driver in Formula One history at the Australian Grand Prix. Following several points finishes in his debut season, Verstappen retained his seat for 2016 before being promoted to parent team Red Bull after four rounds. On debut for Red Bull, aged 18, Verstappen won the Spanish Grand Prix, becoming the youngest-ever driver to win a Formula One Grand Prix. Verstappen achieved multiple race wins in his 2017 and 2018 campaigns, before finishing third in both the 2019 and 2020 World Drivers' Championships under Honda power.

    Verstappen won his maiden title in 2021 after overtaking Lewis Hamilton on the final lap of the last race of the season, becoming the first World Drivers' Champion from the Netherlands. Verstappen won the next two championships in 2022 and 2023, overturning the largest points deficit in Formula One history in the former and breaking numerous records across both seasons.[c] He secured his fourth consecutive title in 2024 after winning nine Grands Prix, including a widely acclaimed wet-weather performance in São Paulo, to become the first driver to win the championship driving for a third-placed constructor in 41 years.

    As of the 2025 Azerbaijan Grand Prix, Verstappen has achieved 67 race wins, 46 pole positions, 35 fastest laps, and 120 podiums in Formula One. In addition to being the youngest Grand Prix winner, he holds several Formula One records, including the most wins in a season (19), the most podium finishes in a season (21), the most consecutive wins (10), and the most consecutive pole positions (8, shared with Ayrton Senna). Verstappen is contracted to remain at Red Bull until at least the end of the 2028 season.[1] He has also competed professionally in sim racing since 2015, winning several marquee iRacing events. Verstappen was listed in the 2024 issue of Time as one of the 100 most influential people globally, and was appointed an Officer of the Order of Orange-Nassau in 2022.
    """
    summary_template = """
    given the infomation {information} about a person I want you to create:
    1.short summary
    2.two interesting fact about them.
    """
    summary_prompt_template = PromptTemplate(
        input_variables= ["information"],template=summary_template
    )

    llm = OllamaLLM(model="llama3.2:latest",temperature=0)
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information":information})
    print(response)

if __name__ == "__main__":
    main()
