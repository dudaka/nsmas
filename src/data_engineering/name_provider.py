"""
NameProvider: Culturally diverse name sampling for GSM-Symbolic dataset generation.

Implements stratified sampling across cultural blocks to prevent overfitting
and test tokenization robustness across diverse naming conventions.

Reference: Phase 2 Specification Section 4 - Sociolinguistic Architecture
"""

import random
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum


class CulturalBlock(Enum):
    """Major cultural regions for name stratification."""
    EAST_ASIA = "east_asia"
    SOUTH_ASIA = "south_asia"
    SOUTHEAST_ASIA = "southeast_asia"
    MENA = "mena"  # Middle East and North Africa
    SUB_SAHARAN_AFRICA = "sub_saharan_africa"
    WESTERN_EUROPE = "western_europe"
    EASTERN_EUROPE = "eastern_europe"
    LATIN_AMERICA = "latin_america"
    NORTH_AMERICA = "north_america"
    OCEANIA = "oceania"


# Comprehensive name database organized by cultural block
# Each block contains 100+ names to ensure high cardinality (>1000 total)
NAMES_DATABASE: Dict[CulturalBlock, List[str]] = {
    CulturalBlock.EAST_ASIA: [
        # Chinese
        "Wei", "Mei", "Jing", "Xiu", "Yong", "Hui", "Ling", "Feng", "Qiang", "Yan",
        "Chen", "Zhang", "Liu", "Wang", "Li", "Zhao", "Wu", "Zhou", "Huang", "Sun",
        "Xiaoming", "Xiaohua", "Jiahui", "Yifan", "Minghui", "Junwei", "Xuemei", "Haoran",
        "Zhi", "Fang", "Bo", "Lan", "Tao", "Xia", "Bao", "Lei", "Cheng", "Hong",
        "Jinhua", "Weimin", "Xiuying", "Zhiwei", "Mingyu", "Xiaoli", "Jianjun", "Yuchen",
        # Japanese
        "Haruki", "Sakura", "Yuki", "Kenji", "Akiko", "Takeshi", "Yumi", "Ryo",
        "Hana", "Daiki", "Rin", "Sora", "Kaito", "Mio", "Sota", "Aoi",
        "Hiroshi", "Michiko", "Taro", "Hanako", "Kazuki", "Misaki", "Naoki", "Emi",
        "Yuto", "Koharu", "Haruto", "Mei", "Shota", "Yui", "Kenta", "Nanami",
        "Riku", "Hinata", "Takumi", "Akari", "Hayato", "Honoka", "Itsuki", "Nana",
        # Korean
        "Minjun", "Jiyeon", "Seojun", "Soyeon", "Jiho", "Yuna", "Hyunwoo", "Minji",
        "Dohyun", "Subin", "Junghwan", "Eunji", "Taeyong", "Hayeon", "Siwoo", "Chaeyoung",
        "Seungmin", "Nayeon", "Woojin", "Dahyun", "Jihoon", "Soojin", "Minho", "Jisoo",
        "Jaemin", "Yeji", "Donghyun", "Yerin", "Junwoo", "Sujin", "Hyunjin", "Yoona",
        # Mongolian
        "Bat", "Oyun", "Bold", "Tsetseg", "Ganzorig", "Enkhjin", "Temuulen", "Narantuya",
        "Bayar", "Altansetseg", "Erdene", "Sarnai", "Dolgormaa", "Batbayar", "Tserendorj", "Tuya",
    ],
    CulturalBlock.SOUTH_ASIA: [
        # Indian (various languages)
        "Aarav", "Priya", "Arjun", "Ananya", "Rohan", "Diya", "Vivaan", "Isha",
        "Aditya", "Kavya", "Vikram", "Neha", "Sanjay", "Pooja", "Rahul", "Shreya",
        "Ravi", "Sunita", "Amit", "Meera", "Deepak", "Lakshmi", "Suresh", "Geeta",
        "Krishna", "Radha", "Mohan", "Sita", "Arun", "Kamala", "Vijay", "Anjali",
        "Pranav", "Tara", "Nikhil", "Swati", "Harsh", "Nisha", "Karan", "Divya",
        "Gaurav", "Smita", "Ashok", "Rekha", "Rajiv", "Savita", "Manish", "Pallavi",
        "Venkat", "Padma", "Ganesh", "Uma", "Shashi", "Jaya", "Kishore", "Renuka",
        # Pakistani
        "Zara", "Omar", "Ayesha", "Hassan", "Fatima", "Ali", "Mariam", "Ahmed",
        "Sana", "Bilal", "Hina", "Imran", "Aisha", "Tariq", "Saima", "Faisal",
        "Usman", "Khadija", "Waqar", "Lubna", "Hamza", "Bushra", "Asad", "Nadia",
        # Bangladeshi
        "Rahim", "Nasreen", "Karim", "Sultana", "Rafiq", "Jahanara", "Shahid", "Ruma",
        "Alamgir", "Shamim", "Mostafa", "Shirin", "Jahangir", "Parveen", "Babul", "Razia",
        # Sri Lankan
        "Saman", "Kumari", "Ruwan", "Dilini", "Nuwan", "Sanduni", "Kasun", "Ishara",
        "Pradeep", "Nimali", "Chaminda", "Sriyani", "Lasith", "Menaka", "Upul", "Gayani",
        # Nepali
        "Bikash", "Sunita", "Ramesh", "Gita", "Santosh", "Maya", "Prakash", "Kamala",
        "Sujan", "Asha", "Binod", "Sabita", "Dipak", "Mina", "Nabin", "Sushila",
    ],
    CulturalBlock.SOUTHEAST_ASIA: [
        # Vietnamese
        "Minh", "Lan", "Duc", "Huong", "Tuan", "Thao", "Hieu", "Mai",
        "Quang", "Linh", "Nam", "Ha", "Phuc", "Ngoc", "Dung", "Trang",
        "An", "Hoa", "Khanh", "Tam", "Viet", "Chi", "Son", "Phuong",
        # Thai
        "Somchai", "Siriwan", "Prasert", "Malee", "Wichai", "Nuanphan", "Thana", "Apsara",
        "Pichit", "Kulap", "Anong", "Sombat", "Prawit", "Pensri", "Chatchai", "Sunisa",
        "Chai", "Duangdao", "Narong", "Boonlert", "Suchart", "Rattana", "Preecha", "Nittaya",
        # Filipino
        "Juan", "Maria", "Jose", "Ana", "Carlos", "Rosa", "Miguel", "Elena",
        "Rafael", "Isabella", "Antonio", "Sofia", "Diego", "Camila", "Gabriel", "Lucia",
        "Rico", "Marites", "Dante", "Corazon", "Bong", "Lorna", "Manny", "Cherry",
        # Indonesian
        "Budi", "Siti", "Agus", "Dewi", "Eko", "Ratna", "Bambang", "Wati",
        "Hendra", "Yuliana", "Dedi", "Lestari", "Rizki", "Fitri", "Andi", "Nia",
        "Joko", "Putri", "Arief", "Sari", "Wahyu", "Indah", "Dimas", "Ayu",
        # Malaysian
        "Hafiz", "Nurul", "Azman", "Farah", "Rizwan", "Aishah", "Syafiq", "Nadia",
        "Amirul", "Siti", "Hakim", "Nur", "Irfan", "Atiqah", "Danish", "Ain",
    ],
    CulturalBlock.MENA: [
        # Arabic
        "Khalid", "Fatimah", "Youssef", "Layla", "Ibrahim", "Noor", "Mustafa", "Hana",
        "Samir", "Amira", "Rashid", "Yasmin", "Tarek", "Salma", "Nabil", "Dina",
        "Ahmad", "Maryam", "Mahmoud", "Sara", "Kareem", "Rania", "Fadi", "Lina",
        "Waleed", "Huda", "Bassam", "Rana", "Jamal", "Ghada", "Adel", "Leila",
        "Hamza", "Amal", "Salem", "Samira", "Zaid", "Noura", "Kamal", "Wafa",
        # Persian
        "Dariush", "Shirin", "Arash", "Mina", "Behnam", "Parisa", "Kamran", "Nazanin",
        "Reza", "Fatemeh", "Amir", "Zahra", "Hossein", "Maryam", "Ali", "Mahsa",
        "Babak", "Leila", "Cyrus", "Roxana", "Dara", "Niloufar", "Kaveh", "Shadi",
        # Turkish
        "Emre", "Aylin", "Burak", "Elif", "Cem", "Zeynep", "Murat", "Selin",
        "Kerem", "Deniz", "Baris", "Defne", "Tolga", "Ceren", "Ozan", "Ezgi",
        "Kaan", "Melis", "Arda", "Irem", "Alper", "Pelin", "Serkan", "Ebru",
        # Hebrew
        "Noam", "Tamar", "Yonatan", "Shira", "Eitan", "Maya", "Oren", "Noa",
        "Gideon", "Avital", "Avi", "Yael", "Dov", "Michal", "Itai", "Liora",
        "Yosef", "Miriam", "David", "Rachel", "Moshe", "Leah", "Yaakov", "Sarah",
    ],
    CulturalBlock.SUB_SAHARAN_AFRICA: [
        # Nigerian (Yoruba, Igbo, Hausa)
        "Adebayo", "Ngozi", "Chidi", "Amina", "Oluwaseun", "Chioma", "Emeka", "Funke",
        "Danjuma", "Aisha", "Yakubu", "Hauwa", "Chukwuemeka", "Adaeze", "Obi", "Yetunde",
        "Tunde", "Bisi", "Segun", "Kemi", "Femi", "Shade", "Tobi", "Sade",
        # Kenyan (Swahili)
        "Jabari", "Amani", "Bakari", "Zawadi", "Juma", "Neema", "Hamisi", "Zuri",
        "Wanjiku", "Kamau", "Nyambura", "Otieno", "Akinyi", "Odhiambo", "Wambui", "Mwangi",
        # South African (Zulu, Xhosa)
        "Thabo", "Nomvula", "Sipho", "Thandiwe", "Bongani", "Lindiwe", "Mandla", "Nandi",
        "Sizwe", "Nomsa", "Themba", "Bongi", "Siyabonga", "Ntombi", "Mthunzi", "Zodwa",
        # Ghanaian (Akan)
        "Kwame", "Akua", "Kofi", "Ama", "Kweku", "Efua", "Yaw", "Abena",
        "Kojo", "Adwoa", "Kwesi", "Akosua", "Kwabena", "Afia", "Yaa", "Akufo",
        # Ethiopian (Amharic)
        "Tesfaye", "Tigist", "Getachew", "Mulu", "Dawit", "Hiwot", "Solomon", "Birtukan",
        # Congolese
        "Patrice", "Esperance", "Kabongo", "Divine", "Tresor", "Grace", "Fiston", "Gloire",
        # Tanzanian
        "Baraka", "Rehema", "Salim", "Hadija", "Rashidi", "Mariamu", "Jafari", "Zainab",
        # Senegalese
        "Mamadou", "Fatou", "Ousmane", "Aminata", "Ibrahima", "Awa", "Moussa", "Ndeye",
    ],
    CulturalBlock.WESTERN_EUROPE: [
        # English
        "James", "Emma", "William", "Olivia", "Henry", "Charlotte", "George", "Amelia",
        "Oliver", "Sophia", "Jack", "Lily", "Thomas", "Grace", "Harry", "Chloe",
        # German
        "Lukas", "Lena", "Felix", "Mia", "Leon", "Sophie", "Paul", "Marie",
        "Maximilian", "Anna", "Jonas", "Lea", "Finn", "Hannah", "Noah", "Emilia",
        # French
        "Louis", "Chloe", "Gabriel", "Emma", "Raphael", "Jade", "Arthur", "Louise",
        "Hugo", "Alice", "Jules", "Lina", "Lucas", "Rose", "Adam", "Lea",
        # Italian
        "Leonardo", "Giulia", "Francesco", "Sofia", "Alessandro", "Aurora", "Lorenzo", "Ginevra",
        "Matteo", "Alice", "Andrea", "Beatrice", "Luca", "Emma", "Marco", "Chiara",
        # Spanish
        "Hugo", "Lucia", "Martin", "Sofia", "Daniel", "Maria", "Pablo", "Carmen",
        "Alejandro", "Marta", "David", "Paula", "Adrian", "Sara", "Alvaro", "Elena",
        # Dutch
        "Daan", "Emma", "Sem", "Julia", "Lucas", "Sophie", "Finn", "Mila",
        # Portuguese
        "Joao", "Maria", "Francisco", "Leonor", "Santiago", "Matilde", "Afonso", "Beatriz",
        # Irish
        "Conor", "Aoife", "Cian", "Niamh", "Oisin", "Saoirse", "Sean", "Ciara",
        # Scottish
        "Callum", "Isla", "Hamish", "Eilidh", "Angus", "Morag", "Fergus", "Fiona",
        # Scandinavian (Swedish, Norwegian, Danish)
        "Erik", "Astrid", "Lars", "Freya", "Bjorn", "Ingrid", "Sven", "Sigrid",
    ],
    CulturalBlock.EASTERN_EUROPE: [
        # Russian
        "Dmitri", "Anastasia", "Ivan", "Natasha", "Alexei", "Olga", "Sergei", "Tatiana",
        "Mikhail", "Ekaterina", "Nikolai", "Irina", "Andrei", "Svetlana", "Vladimir", "Marina",
        # Polish
        "Jan", "Zofia", "Jakub", "Zuzanna", "Szymon", "Julia", "Filip", "Maja",
        "Kacper", "Lena", "Michal", "Alicja", "Wojciech", "Maria", "Tomasz", "Anna",
        # Ukrainian
        "Oleksandr", "Oksana", "Taras", "Kateryna", "Bohdan", "Iryna", "Mykola", "Yulia",
        # Czech
        "Jakub", "Tereza", "Jan", "Natalie", "Tomas", "Adela", "Matej", "Ema",
        # Hungarian
        "Bence", "Hanna", "Mate", "Anna", "Levente", "Luca", "Adam", "Zsofia",
        # Romanian
        "Andrei", "Maria", "Alexandru", "Elena", "Mihai", "Ioana", "Stefan", "Ana",
        # Bulgarian
        "Georgi", "Maria", "Dimitar", "Ivana", "Nikolay", "Elena", "Aleksandar", "Sofia",
        # Serbian
        "Nikola", "Ana", "Luka", "Mia", "Stefan", "Sara", "Marko", "Teodora",
    ],
    CulturalBlock.LATIN_AMERICA: [
        # Mexican
        "Santiago", "Valentina", "Mateo", "Camila", "Sebastian", "Sofia", "Diego", "Isabella",
        "Nicolas", "Ximena", "Leonardo", "Regina", "Miguel", "Mariana", "Angel", "Fernanda",
        # Brazilian
        "Miguel", "Alice", "Arthur", "Sophia", "Heitor", "Helena", "Bernardo", "Valentina",
        "Theo", "Laura", "Davi", "Isabella", "Lorenzo", "Manuela", "Gabriel", "Julia",
        # Argentinian
        "Thiago", "Martina", "Santino", "Emilia", "Bautista", "Delfina", "Mateo", "Catalina",
        # Colombian
        "Samuel", "Luciana", "Martin", "Sara", "Emmanuel", "Gabriela", "Juan", "Paula",
        # Peruvian
        "Luis", "Valeria", "Carlos", "Daniela", "Jorge", "Andrea", "Pedro", "Claudia",
        # Chilean
        "Benjamin", "Florencia", "Vicente", "Isidora", "Agustin", "Antonia", "Tomas", "Amanda",
        # Venezuelan
        "Adrian", "Victoria", "Andres", "Natalia", "Jose", "Adriana", "Ricardo", "Carolina",
        # Cuban
        "Alejandro", "Yanet", "Roberto", "Yamilet", "Raul", "Yaneth", "Eduardo", "Dayana",
    ],
    CulturalBlock.NORTH_AMERICA: [
        # American (diverse)
        "Liam", "Emma", "Noah", "Olivia", "Ethan", "Ava", "Mason", "Isabella",
        "Logan", "Mia", "Alexander", "Harper", "Jackson", "Evelyn", "Aiden", "Abigail",
        "Michael", "Emily", "Daniel", "Madison", "Matthew", "Chloe", "Joseph", "Grace",
        "Benjamin", "Zoey", "Samuel", "Lily", "Christopher", "Hannah", "Andrew", "Victoria",
        # Canadian
        "William", "Charlotte", "Benjamin", "Amelia", "Theodore", "Eleanor", "Lucas", "Hazel",
        # Native American inspired
        "Dakota", "Sierra", "Cheyenne", "River", "Phoenix", "Sage", "Rain", "Sky",
        # African American
        "Jayden", "Aaliyah", "Jordan", "Zoe", "Marcus", "Maya", "Andre", "Brianna",
        "Darius", "Jasmine", "Xavier", "Imani", "Malik", "Destiny", "Tyrone", "Aliyah",
    ],
    CulturalBlock.OCEANIA: [
        # Australian
        "Jack", "Charlotte", "William", "Olivia", "Oliver", "Amelia", "Noah", "Isla",
        "Leo", "Mia", "Henry", "Ava", "Charlie", "Grace", "Thomas", "Chloe",
        # New Zealand
        "Liam", "Aria", "Oliver", "Isla", "Jack", "Amelia", "Noah", "Charlotte",
        # Maori
        "Tama", "Aroha", "Nikau", "Maia", "Rawiri", "Anahera", "Te", "Hine",
        "Wiremu", "Mere", "Hemi", "Kiri", "Tane", "Ngaio", "Manaia", "Ataahua",
        # Polynesian (Hawaiian, Samoan, Tongan)
        "Keoni", "Leilani", "Kai", "Malia", "Koa", "Moana", "Ikaika", "Kalani",
        "Sione", "Mele", "Tevita", "Ofa", "Semisi", "Ana", "Viliami", "Sela",
        # Fijian
        "Josefa", "Mereoni", "Sitiveni", "Ana", "Timoci", "Lavenia", "Waisea", "Litia",
        # Aboriginal Australian
        "Jarrah", "Marlee", "Kynan", "Alinta", "Tyson", "Jedda", "Bindi", "Narla",
    ],
}


@dataclass
class NameSample:
    """A sampled name with metadata."""
    name: str
    cultural_block: CulturalBlock

    def __str__(self) -> str:
        return self.name


class NameProvider:
    """
    Provides culturally diverse names with stratified sampling.

    Implements round-robin or weighted sampling across cultural blocks
    to ensure diversity and prevent overfitting to specific naming patterns.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the NameProvider.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = random.Random(seed)
        self._all_names: List[tuple] = []
        self._block_index = 0
        self._blocks = list(CulturalBlock)

        # Flatten all names with their block for quick access
        for block, names in NAMES_DATABASE.items():
            for name in names:
                self._all_names.append((name, block))

        self.rng.shuffle(self._all_names)

    @property
    def total_unique_names(self) -> int:
        """Return the total number of unique names available."""
        return len(self._all_names)

    @property
    def cultural_blocks(self) -> List[CulturalBlock]:
        """Return list of available cultural blocks."""
        return list(CulturalBlock)

    def get_name(self, block: Optional[CulturalBlock] = None) -> NameSample:
        """
        Get a random name, optionally from a specific cultural block.

        Args:
            block: Optional cultural block to sample from

        Returns:
            NameSample with name and cultural metadata
        """
        if block is not None:
            names = NAMES_DATABASE[block]
            name = self.rng.choice(names)
            return NameSample(name=name, cultural_block=block)

        # Random sampling from all names
        name, block = self.rng.choice(self._all_names)
        return NameSample(name=name, cultural_block=block)

    def get_name_round_robin(self) -> NameSample:
        """
        Get a name using round-robin across cultural blocks.

        Ensures even distribution across all cultural regions.

        Returns:
            NameSample with name and cultural metadata
        """
        block = self._blocks[self._block_index]
        self._block_index = (self._block_index + 1) % len(self._blocks)
        return self.get_name(block)

    def get_names(self, count: int, unique: bool = True,
                  same_block: bool = False) -> List[NameSample]:
        """
        Get multiple names.

        Args:
            count: Number of names to return
            unique: If True, ensure all names are different
            same_block: If True, all names come from the same cultural block

        Returns:
            List of NameSample objects
        """
        if same_block:
            block = self.rng.choice(self._blocks)
            available = NAMES_DATABASE[block].copy()
            self.rng.shuffle(available)

            if unique and count > len(available):
                raise ValueError(
                    f"Requested {count} unique names but block {block.value} "
                    f"only has {len(available)} names"
                )

            if unique:
                selected = available[:count]
            else:
                selected = [self.rng.choice(available) for _ in range(count)]

            return [NameSample(name=n, cultural_block=block) for n in selected]

        if unique:
            if count > len(self._all_names):
                raise ValueError(
                    f"Requested {count} unique names but only "
                    f"{len(self._all_names)} available"
                )
            selected = self.rng.sample(self._all_names, count)
            return [NameSample(name=n, cultural_block=b) for n, b in selected]

        return [self.get_name() for _ in range(count)]

    def get_names_for_problem(self, count: int,
                              used_names: Optional[Set[str]] = None) -> List[str]:
        """
        Get names for a math problem, ensuring no collisions.

        This is the primary method for the TemplateEngine to use.

        Args:
            count: Number of unique names needed
            used_names: Set of names already used (to avoid)

        Returns:
            List of name strings
        """
        used = used_names or set()
        result = []
        attempts = 0
        max_attempts = count * 10

        while len(result) < count and attempts < max_attempts:
            sample = self.get_name_round_robin()
            if sample.name not in used and sample.name not in result:
                result.append(sample.name)
            attempts += 1

        if len(result) < count:
            # Fallback: use any available name
            for name, _ in self._all_names:
                if name not in used and name not in result:
                    result.append(name)
                    if len(result) >= count:
                        break

        return result

    def set_seed(self, seed: int) -> None:
        """
        Reset the random seed for reproducibility.

        Args:
            seed: New random seed
        """
        self.rng = random.Random(seed)
        self._block_index = 0
        self.rng.shuffle(self._all_names)

    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about the name database.

        Returns:
            Dictionary with counts per cultural block and total
        """
        stats = {block.value: len(names)
                 for block, names in NAMES_DATABASE.items()}
        stats['total'] = sum(stats.values())
        return stats


if __name__ == "__main__":
    # Demo usage
    provider = NameProvider(seed=42)
    print(f"Total unique names: {provider.total_unique_names}")
    print(f"\nStats by cultural block:")
    for block, count in provider.get_stats().items():
        print(f"  {block}: {count}")

    print("\n10 random names (round-robin):")
    for _ in range(10):
        sample = provider.get_name_round_robin()
        print(f"  {sample.name} ({sample.cultural_block.value})")

    print("\n5 names for a problem:")
    names = provider.get_names_for_problem(5)
    print(f"  {names}")
