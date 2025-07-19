import torch
from transformers import pipeline
import requests
import json
from lexical_chains_spacy import build_similarity_chains, select_chain_sentences

from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

import tkinter as tk
import tkinter.ttk as ttk
import yaml
from tkinter import filedialog, scrolledtext, messagebox
import subprocess
import os
import pandas as pd
import xmltodict
import json
import re
import spacy
import nltk
import webbrowser
import threading
from nltk.parse.stanford import StanfordParser
from nltk.tree import Tree
from nltk.tokenize import sent_tokenize
from nltk.parse import stanford
from spacy import displacy
import requests
from bs4 import BeautifulSoup
import stanza
import spacy_stanza
from negspacy.negation import Negex
from negspacy.termsets import termset


class TextViewer:
    def load_config(config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
    config = load_config('config.yaml')
    # set the variables from the configuration
    installation_dir = config['ctakes']['installation_dir']
    input_dir = config['ctakes']['input_dir']
    output_dir = config['ctakes']['output_dir']
    pipeline_key = config['ctakes']['pipeline_key']
    stanford_parser_path_to_jar = config['stanford_parser']['path_to_jar']
    stanford_parser_path_to_models_jar = config['stanford_parser']['path_to_models_jar']
    
    def __init__(self, root):
        # Setting the title of the GUI
        self.root = root
        self.root.title("Medical Text Ontology Analyzer")
        # Create a weblink paste area widget
        self.web_link_are_label = tk.Label(self.root, text="Paste a web link:")
        self.web_link_are_label.pack(pady=5)
        self.link_entry = tk.Entry(self.root, width=70)
        self.link_entry.pack(pady=10)

        self.clinical_entities =set()
        # create link frame Buttons
        link_frame = tk.Frame(self.root)
        link_frame.pack(pady=10)
        link_buttons = [
            ("Extract text from the link", self.process_link),
            ("?", self.show_extract_text_help)
        ]
        for text, command in link_buttons:
            btn = tk.Button(link_frame, text=text, command=command)
            btn.pack(side=tk.LEFT, padx=10)
        # Create a scrolled text area widget
        self.text_area_label = tk.Label(self.root, text="Enter text:")
        self.text_area_label.pack()
        self.text_area = scrolledtext.ScrolledText(self.root, height = 10, width = 200, wrap=tk.WORD)
        self.text_area.pack()
        # create Text frame Buttons
        text_frame = tk.Frame(self.root)
        text_frame.pack(pady=10)
        text_buttons = [
            ("Browse text file", self.open_file),
            ("Run cTakes Clinical Pipeline", self.run_clinical_pipeline),
            ("?", self.Run_Clinical_Pipeline_help),
            ("Clear Text", self.clear_viewer)
        ]
        for text, command in text_buttons:
            btn = tk.Button(text_frame, text=text, command=command)
            btn.pack(side=tk.LEFT, padx=10)
        # Separator Line
        separator = ttk.Separator(root, orient="horizontal")
        separator.pack(fill="x", padx=5, pady=5)
        # Create an output viewer to display the output content
        self.output_viewer_label = tk.Label(self.root, text="Output:")
        self.output_viewer_label.pack()
        self.output_viewer = scrolledtext.ScrolledText(self.root, height=10, width = 200, wrap=tk.WORD)
        self.output_viewer.pack()
        # General NLP Label
        general_nlp_label = tk.Label(self.root, text="General NLP Analysis", font=("Helvetica", 10, "bold"))
        general_nlp_label.pack()
        # General NLP Analysis Buttons
        self.current_sentence_index = 0
        general_nlp_frame = tk.Frame(self.root)
        general_nlp_frame.pack(pady=2)
        general_nlp_buttons = [
            ("General Name Entity", self.process_general_NE),
            ("Chunk Parsing", self.chunk_parsing),
            ("Constituency parsing", self.constituency_parsing),
            ("Dependency Parsing", self.dependency_parsing),
            ("Semantic Role Labeling", self.semantic_role_labeling)
        ]
        for text, command in general_nlp_buttons:
            btn = tk.Button(general_nlp_frame, text=text, command=command)
            btn.pack(side=tk.LEFT, padx=5)
        general_nlp_help_frame = tk.Frame(self.root)
        general_nlp_help_frame.pack(pady=1)
        general_nlp_help_buttons = [
            ("?", self.process_general_NE_help),
            ("?", self.chunk_parsing_help),
            ("?", self.constituency_parsing_help),
            ("?", self.dependency_parsing_help),
            ("?", self.semantic_role_labeling_help),
        ]
        for text, command in general_nlp_help_buttons:
            btn = tk.Button(general_nlp_help_frame, text=text, command=command)
            btn.pack(side=tk.LEFT, padx=50)
        # Separator Line
        separator = ttk.Separator(root, orient="horizontal")
        separator.pack(fill="x", padx=5, pady=5)
        # Medical NLP Label
        medical_nlp_label = tk.Label(self.root, text="Medical NLP Analysis", font=("Helvetica", 10, "bold"))
        medical_nlp_label.pack(pady=5)
        # Medical NLP Buttons
        medical_nlp_frame = tk.Frame(self.root)
        medical_nlp_frame.pack(pady=10)
        medical_nlp_buttons = [
            ("Clinical Name Entity", self.process_clinical_NE),
            ("Clinical Negation", self.negation),
            ("Clinical Polarity", self.polarity),
            ("Summarize", self.summarize_text_lexrank),
           
        ]
        for text, command in medical_nlp_buttons:
            btn = tk.Button(medical_nlp_frame, text=text, command=command)
            btn.pack(side=tk.LEFT, padx=5)
        medical_nlp_help_frame = tk.Frame(self.root)
        medical_nlp_help_frame.pack(pady=1)
        medical_nlp_help_buttons = [
            ("?", self.process_clinical_NE_help),
            ("?", self.negation_help),
            ("?", self.polarity_help),
            ("?", self.summarize_text_lexrank_help),
            
        ]
        for text, command in medical_nlp_help_buttons:
            btn = tk.Button(medical_nlp_help_frame, text=text, command=command)
            btn.pack(side=tk.LEFT, padx=50)
        
      
        



#########################################################################################################################
      
    
        
      
     

    # Clinical entities extraction 
    def get_clinical_entities(self):
        if not self.clinical_entities:
            result_df = self.data_processing()
            self.clinical_entities = set(result_df['Entity Name'].dropna().unique())
        return self.clinical_entities



    
        
    # Perform LexRank summarization
    def summarize_text_lexrank(self):
        text = self.text_area.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("No Text", "No text input found.")
            return
        
        # 1) cTAKES → clinical entity sentences
        sentences = sent_tokenize(text)
        clinical = []
        for i, sent in enumerate(sentences):
            if any(ent.lower() in sent.lower() for ent in self.get_clinical_entities()):
                clinical.append(sent)
                if i>0:   clinical.append(sentences[i-1])
                if i<len(sentences)-1: clinical.append(sentences[i+1])
        clinical = list(dict.fromkeys(clinical))
         # Step 2: build SpaCy similarity chains & pick extra sentences
        chains = build_similarity_chains(sentences, top_k=5)
        extra_idxs = select_chain_sentences(chains, per_chain=2)
        extra_sents = [sentences[i] for i in extra_idxs if sentences[i] not in clinical]
        # 3) Merge & dedupe in original order
        candidates = clinical + extra_sents
        ordered = list(dict.fromkeys(candidates))

        # 4) Run LexRank
        parser = PlaintextParser.from_string(" ".join(ordered), Tokenizer("english"))
        summarizer = LexRankSummarizer()
        summary = summarizer(parser.document, sentences_count=min(10, len(ordered)))
        # 5) Display
        self.output_viewer.delete("1.0", tk.END)
        # self.output_viewer.insert(tk.END, "LexRank+Chains Summary:\n\n")
        for sent in summary:
            self.output_viewer.insert(tk.END, str(sent) + "\n")


    # Add this method to provide help for LexRank summarization
    def summarize_text_lexrank_help(self):
        help_text ="""Click "Summarize Text (LexRank)" to generate a summary of the input text using the LexRank algorithm.
        The summary will be displayed in the output viewer."""
        messagebox.showinfo("Help", help_text) 
        
       


        
        
#########################################################################################################################
    def open_file(self):
        # Open a file dialog to prompt the user to select a file
        file_path = filedialog.askopenfilename()
        if file_path:
            with open(file_path, "r") as file:
                text_content = file.read()
            self.text_area.delete("1.0", tk.END)
            self.text_area.insert(tk.END, text_content)
    def clear_viewer(self):
        # clear the text and output area
        self.text_area.delete("1.0", tk.END)
        self.output_viewer.delete("1.0", tk.END)
    def run_clinical_pipeline(self):
        # Get the text from the text area
        text = self.text_area.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("No Text", "No text input found.")
            return
        # Save the text to a temporary file
        temp_file = f'{self.input_dir}\\temp_input'
        with open(temp_file, "w") as file:
            # Write the text content to the temporary file
            file.write(text)
        try:
            # Call the cTakes clinical pipeline to process the text
            command = f'{self.installation_dir}\\bin\\runClinicalPipeline -i {self.input_dir} --xmiOut {self.output_dir} --key {self.pipeline_key}'
            # Run the command as a subprocess, using the cTakes directory as the current working directory
            subprocess.run(command, shell=True, cwd=f'{self.installation_dir}')
            # Delete the temporary input file
            os.remove(temp_file)
            # Show a success message indicating that the text was processed successfully
            messagebox.showinfo("Text Processed", "Text processed successfully!")
        except subprocess.CalledProcessError:
            # Display an error message if an error occurs during text processing
            messagebox.showerror("Error", "An error occurred while processing the text.")
    def process_clinical_NE(self):
        # Clear the output viewer
        self.output_viewer.delete('1.0', tk.END)
        # Call the data_processing function
        result_df = self.data_processing()
     
        
        for _, row in result_df.iterrows():
            entity_type = row['Entity Type']
            entity_name = row['Entity Name']
            tag = entity_type.lower()
            self.output_viewer.insert(tk.END, f"{entity_type}: {entity_name}\n", tag)
        self.output_viewer.tag_config('disease_disorder_mention', foreground='red')
        self.output_viewer.tag_config('medication_mention', foreground='blue')
        self.output_viewer.tag_config('anatomical_site_mention', foreground='green')
        self.output_viewer.tag_config('procedure_mention', foreground='purple')
        self.output_viewer.tag_config('sign_symptom_mention', foreground='orange')
        
    def create_dataframe(self,data, data_name):
        df_data = []
        if isinstance(data, dict):
            df_data.append({'Entity Type': data_name, '@begin': data.get('@begin'), '@end': data.get('@end')})
        elif isinstance(data, list):
            for item in data:
                df_data.append({'Entity Type': data_name, '@begin': item.get('@begin'), '@end': item.get('@end')})
        df = pd.DataFrame(df_data)
        return df
    def append_normalized_form(self,df, data):
        normalized_forms = []
        for _, row in df.iterrows():
            begin = int(row['@begin'])
            normalized_form = None
            try:
                for item in data:
                    item_begin = int(item['@begin'])
                    
                    if begin == item_begin:
                        normalized_form = item['@form']
                        break
            except KeyError:
                pass
            normalized_forms.append(normalized_form)
        df['Entity Name'] = normalized_forms
        return df
    def data_processing(self):
        with open(f'{self.output_dir}temp_input.xmi') as xml_file:
            data_dict = xmltodict.parse(xml_file.read())
        json_data = json.dumps(data_dict)
        with open("data_1.json", "w") as json_file:
            json_file.write(json_data)
        filename = "data_1.json"
        with open(filename, "r") as file:
            data = json.load(file)
        try:
            Medication_Mention = data['xmi:XMI']['textsem:MedicationMention']
        except KeyError:
            Medication_Mention = ''
        try:
            Procedure_Mention = data['xmi:XMI']['textsem:ProcedureMention']
        except KeyError:
            Procedure_Mention = ''
        try:
            Sign_Symptom_Mention = data['xmi:XMI']['textsem:SignSymptomMention']
        except KeyError:
            Sign_Symptom_Mention = ''
        try:
            Disease_Disorder_Mention = data['xmi:XMI']['textsem:DiseaseDisorderMention']
        except KeyError:
            Disease_Disorder_Mention = ''
        try:
            Anatomical_Site_Mention = data['xmi:XMI']['textsem:AnatomicalSiteMention']
        except KeyError:
            Anatomical_Site_Mention = ''
        df_MM = self.create_dataframe(Medication_Mention, 'Medication_Mention')
        df_PM = self.create_dataframe(Procedure_Mention, 'Procedure_Mention')
        df_SSM = self.create_dataframe(Sign_Symptom_Mention, 'Sign_Symptom_Mention')
        df_DDM = self.create_dataframe(Disease_Disorder_Mention, 'Disease_Disorder_Mention')
        df_ASM = self.create_dataframe(Anatomical_Site_Mention, 'Anatomical_Site_Mention')
        concat_df = pd.concat([df_DDM, df_MM, df_ASM, df_PM, df_SSM], axis=0)
        input_data = data['xmi:XMI']['syntax:ConllDependencyNode']
        df = self.append_normalized_form(concat_df, input_data)
        df = df[df['Entity Name'].notna()]
        return df[['Entity Type','Entity Name']]
#########################################################################################################################
    def process_general_NE(self):
        # Get the text from the text area
        text = self.text_area.get("1.0", tk.END)
        # Load the spaCy English model
        nlp = spacy.load("en_core_web_sm")
        # Process the text with spaCy
        doc = nlp(text)
        # Initialize lists to store entity types and names
        entity_types = []
        entity_names = []
        # Iterate over the entities in the processed document
        for ent in doc.ents:
            entity_types.append(ent.label_)  # Append the entity type
            entity_names.append(ent.text)    # Append the entity name
        # Create a dataframe from the entity types and names
        entity_df = pd.DataFrame({"Entity Type": entity_types, "Entity Name": entity_names})
        # Clear the output viewer
        self.output_viewer.delete("1.0", tk.END)
        # Display the entity dataframe in the output viewer
        self.output_viewer.insert(tk.END, entity_df.to_string())
#########################################################################################################################
    def chunk_parsing(self):
        text = self.text_area.get("1.0", tk.END).strip()
        sentence_token = sent_tokenize(text)
        if self.current_sentence_index < len(sentence_token):
            sentence = sentence_token[self.current_sentence_index]
            self.current_sentence_index += 1
            self.output_viewer.delete(1.0, tk.END)
            self.output_viewer.insert(tk.END, f"Sentence {self.current_sentence_index}: {sentence}\n\n")
            # Tokenize the sentence into words
            words = nltk.word_tokenize(sentence)
            # Perform Part-of-Speech (POS) tagging
            tagged_words = nltk.pos_tag(words)
            # Define a grammar for chunking
            grammar = r"""
                NP: {<DT|JJ|NN.*>+}          # Chunk sequences of determiners, adjectives, and nouns
                VP: {<VB.*><NP|PP|CLAUSE>+$} # Chunk verbs followed by NP, PP, or CLAUSE
                PP: {<IN><NP>}               # Chunk prepositions followed by NP
                CLAUSE: {<NP><VP>}           # Chunk NP followed by VP
            """
            # Create a chunk parser
            chunk_parser = nltk.RegexpParser(grammar)
            # Perform chunking
            chunks = chunk_parser.parse(tagged_words)
            # Display the result (draw the chunks)
            chunks.draw()
            return chunks
        else:
            self.current_sentence_index = 0
#########################################################################################################################
    def constituency_parsing(self):
        text = self.text_area.get("1.0", tk.END).strip()
        sentence_token = sent_tokenize(text)
        if self.current_sentence_index < len(sentence_token):
            sentence = sentence_token[self.current_sentence_index]
            self.current_sentence_index += 1
            self.output_viewer.delete(1.0, tk.END)
            self.output_viewer.insert(tk.END, f"Sentence {self.current_sentence_index}: {sentence}\n\n")
            parser = stanford.StanfordParser(
            path_to_jar=self.stanford_parser_path_to_jar,
            path_to_models_jar= self.stanford_parser_path_to_models_jar
            )
            # Parse the sentence
            parse_tree = list(parser.raw_parse(sentence))[0]
            # Visualize the parse tree
            parse_tree.draw()
        else:
            self.current_sentence_index = 0
#########################################################################################################################
    def dependency_parsing_processing(self):
        nlp = spacy.load("en_core_web_sm")
        text = self.text_area.get("1.0", tk.END)
        sentence_token = sent_tokenize(text)
        saved_files = []
        dataframe_rows = []
        for sentence in sentence_token:
            doc = nlp(sentence)
            sentence_spans = list(doc.sents)
            options = {"fine_grained": True, "arrow_stroke": 2, "arrow_width": 8}
            svg = displacy.render(sentence_spans, style="dep", options=options, jupyter=False)
            filename = f"Dependency_parsing_sentence_{self.current_sentence_index + 1}.svg"
            saved_files.append(filename)
            with open(filename, "w", encoding="utf-8") as file:
                file.write(svg)
            srl_results = []
            for token in doc:
                srl_result = f"token: {token.text}, relation: {token.dep_}, head: {token.head.text}"
                srl_results.append(srl_result)
            dataframe_row = {'file_name': filename, 'sentence': sentence, 'srl_output': "\n".join(srl_results)}
            dataframe_rows.append(dataframe_row)
            self.current_sentence_index += 1
        dataframe = pd.DataFrame(dataframe_rows)
        return dataframe
    def open_svg_files_in_browser(self, dataframe):
        root = tk.Tk()
        root.title("Dependency parsing Viewer")
        for _, row in dataframe.iterrows():
            file_path = row['file_name']
            srl_output = row['srl_output']
            def display_output_thread(output):
                output_viewer = tk.Toplevel(root)
                output_viewer.title("Dependency parsing Output")
                text = tk.Text(output_viewer)
                text.insert(tk.END, output)
                text.pack()
            open_button = tk.Button(root, text=f"Open {file_path}", command=lambda path=file_path: threading.Thread(target=webbrowser.open(path)).start())
            open_button.pack(pady=5)
            output_button = tk.Button(root, text=f"View relation {file_path}", command=lambda output=srl_output: threading.Thread(target=display_output_thread(output)).start())
            output_button.pack(pady=5)
        root.mainloop()
    def dependency_parsing(self):
        dataframe = self.dependency_parsing_processing()
        self.open_svg_files_in_browser(dataframe)
#########################################################################################################################
    # Function to fill the 'word' column in a DataFrame based on character indices
    def fill_word_column(self, df, begin_col, end_col, text):
        df['word'] = df.apply(lambda row: text[int(row[begin_col]):int(row[end_col])], axis=1)
        return df
    # Function to merge DataFrames based on a key column
    def merge_dataframes_by_key(self, df1, df2, key_col, how='inner'):
        merged_df = df1.merge(df2[[key_col] + list(df2.columns.difference(df1.columns))], on=key_col, how=how)
        return merged_df
    # Main function for semantic role labeling
    def semantic_role_processing(self):
        filename = "data_1.json"
        with open(filename, "r") as file:
            data = json.load(file)
        text = data['xmi:XMI']['cas:Sofa']['@sofaString']
        sentences_df = pd.DataFrame(data['xmi:XMI']['textspan:Sentence'])
        sentences_df = self.fill_word_column(sentences_df, '@begin', '@end', text)
        predicate_df = pd.DataFrame(data['xmi:XMI']['textsem:Predicate'])
        predicate_df = self.fill_word_column(predicate_df, '@begin', '@end', text)
        predicate_df.rename(columns={
            '@xmi:id': 'predicate_id',
            '@sofa': 'subject_of_analysis',
            '@begin': 'starting_character_index',
            '@end': 'ending_character_index',
            '@relation': 'relation',
            '@frameSet': 'frameSet',
            'word': 'predicate_word'
        }, inplace=True)
        semanticargument_df = pd.DataFrame(data['xmi:XMI']['textsem:SemanticArgument'])
        semanticargument_df = self.fill_word_column(semanticargument_df, '@begin', '@end', text)
        semanticargument_df.rename(columns={
            '@xmi:id': 'argument_id',
            '@sofa': 'subject_of_analysis',
            '@begin': 'starting_character_index',
            '@end': 'ending_character_index',
            '@relation': 'relation',
            '@label': 'semantic_role_label',
            'word': 'argument_word'
        }, inplace=True)
        semanticrolerelation_df = pd.DataFrame(data['xmi:XMI']['textsem:SemanticRoleRelation'])
        semanticrolerelation_df.rename(columns={
            '@xmi:id': 'relation',
            '@id': 'id',
            '@category': 'semantic_role_label',
            '@discoveryTechnique': 'discovery_technique',
            '@confidence': 'confidence',
            '@polarity': 'polarity',
            '@uncertainty': 'uncertainty',
            '@conditional': 'conditional',
            '@predicate': 'predicate_id',
            '@argument': 'argument_id',
        }, inplace=True)
        merge_df = self.merge_dataframes_by_key(semanticrolerelation_df, predicate_df[['predicate_id', 'predicate_word']], 'predicate_id')
        merge_df = self.merge_dataframes_by_key(merge_df, semanticargument_df[['starting_character_index', 'ending_character_index', 'argument_id', 'argument_word']], 'argument_id')
        # Convert DataFrame to the desired dictionary format
        semantic_roles = {}
        for _, row in merge_df.iterrows():
            semantic_roles[row['predicate_word']] = "Predicate"
            semantic_roles[row['argument_word']] = row['semantic_role_label']
        return semantic_roles
    # Function to add semantic roles to a parsed tree
    def add_semantic_roles(self,tree_node, roles):
        if tree_node.height() == 2 and tree_node[0] in roles:
            role_label = roles[tree_node[0]]
            new_subtree = Tree(role_label, [tree_node[0]])
            tree_node.clear()
            tree_node.append(new_subtree)
        else:
            for child in tree_node:
                if isinstance(child, Tree):
                    self.add_semantic_roles(child, roles)
    def semantic_role_labeling(self):
        text = self.text_area.get("1.0", tk.END).strip()
        sentence_token = sent_tokenize(text)
        if self.current_sentence_index < len(sentence_token):
            sentence = sentence_token[self.current_sentence_index]
            self.current_sentence_index += 1
            self.output_viewer.delete(1.0, tk.END)
            self.output_viewer.insert(tk.END, f"Sentence {self.current_sentence_index}: {sentence}\n\n")
            parser = stanford.StanfordParser(
            path_to_jar=self.stanford_parser_path_to_jar, 
            path_to_models_jar=self.stanford_parser_path_to_models_jar
            )
            # Parse the sentence
            parsed_result = list(parser.raw_parse(sentence))[0]
            semantic_role = self.semantic_role_processing()
            self.add_semantic_roles(parsed_result, semantic_role)
            parsed_result.draw()
        else:
            self.current_sentence_index = 0
#########################################################################################################################
    def polarity(self):
        with open('data_1.json') as json_file:
            data = json.load(json_file)
        text = data['xmi:XMI']['cas:Sofa']['@sofaString']
        polarity_data = self.extract_polarity_info(data)
        df = pd.DataFrame(polarity_data)
        df = df.dropna()
        df = df.query("polarity != 0")
        polarity_df = self.fill_word_column(df, '@begin', '@end', text)
        self.output_viewer.delete("1.0", tk.END)
        self.output_viewer.insert(tk.END, text)
        # words_to_highlight = polarity_df['word']
        tag_names = ["positive_tag", "neutral_tag", "negative_tag"]
        foreground_colors = ["green", "black", "red"]
        self.highlight_words_from_dataframe(self.output_viewer, polarity_df, tag_names, foreground_colors)
    def extract_polarity_info(self, data):
        polarity_info = []
        if isinstance(data, dict):
            if '@polarity' in data:
                polarity_info.append({
                    'polarity': int(data['@polarity']),
                    '@begin': int(data.get('@begin', 0)),
                    '@end': int(data.get('@end', 0))
                })
            for value in data.values():
                polarity_info.extend(self.extract_polarity_info(value))
        elif isinstance(data, list):
            for item in data:
                polarity_info.extend(self.extract_polarity_info(item))
        return polarity_info
    def highlight_words_from_dataframe(self, text_widget, df, tag_names, foreground_colors, bold=True):
        text_widget.tag_delete("highlight_tag")  # Clear any existing highlight tags
        for polarity, tag_name, foreground_color in zip([1, 0, -1], tag_names, foreground_colors):
            text_widget.tag_configure(tag_name, foreground=foreground_color, font=("Helvetica", 12, "bold" if bold else "normal"))
            polarity_df = df[df['polarity'] == polarity]
            for _, row in polarity_df.iterrows():
                start_index = f"1.{row['@begin']}"
                end_index = f"1.{row['@end']}"
                text_widget.tag_add(tag_name, start_index, end_index)
    #####################################################################################################################
    def negation(self):
        text = self.text_area.get("1.0", tk.END).strip()
        data_frame = self.process_clinical_negation(text)
        self.output_viewer.delete("1.0", tk.END)
        self.output_viewer.insert(tk.END, text)
        # Configure tags for highlighting
        self.output_viewer.tag_config('negated', foreground='red')
        self.output_viewer.tag_config('not_negated', foreground='green')
        self.highlight_entities_with_negated_tags(self.output_viewer, data_frame)
    def process_clinical_negation(self,text):
        nlp = spacy_stanza.load_pipeline('en', package='mimic', processors={'ner': 'i2b2'})
        ts = termset("en_clinical")
        nlp.add_pipe("negex", config={"ent_types":["PROBLEM","TEST",'TREATMENT']})
        doc = nlp(text)
        results = []
        for e in doc.ents:
            results.append({
                'Entity Text': e.text,
                'Entity Type': e.label_,
                'Negation Detected': e._.negex
            })
        df = pd.DataFrame(results)
        return df
    def highlight_entities_with_negated_tags(self, text_widget, entities_df):
        for idx, row in entities_df.iterrows():
            entity_text = row['Entity Text']
            negation_detected = row['Negation Detected']
            if negation_detected:
                tag_name = 'negated'
            else:
                tag_name = 'not_negated'
            start_idx = '1.0'
            while True:
                start_idx = text_widget.search(entity_text, start_idx, stopindex=tk.END)
                if not start_idx:
                    break
                end_idx = f"{start_idx}+{len(entity_text)}c"
                text_widget.tag_add(tag_name, start_idx, end_idx)
                start_idx = end_idx
#########################################################################################################################
    
#########################################################################################################################
    def verify_link(self, link):
        try:
            response = requests.get(link)
            if response.status_code == 200:
                return True
            else:
                return False
        except requests.RequestException:
            return False
    def extract_text(self, link):
            response = requests.get(link)
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text()
            return text
    def process_link(self):
        link = self.link_entry.get()
        if self.verify_link(link):
            extracted_text = self.extract_text(link)
            self.text_area.delete("1.0", tk.END)
            self.text_area.insert(tk.END, extracted_text)
        else:
            messagebox.showerror("Error", "Invalid link or unable to fetch content.")
#########################################################################################################################
    def show_extract_text_help(self):
        help_text = """
        1. Enter a valid link in the designated field.
        2. Click "Extract text from the Link" to process.
        3. Verified links populate relevant content in the text area.
        """
        messagebox.showinfo("Help", help_text)
    def Run_Clinical_Pipeline_help(self):
        help_text = """Click "Run Ctakes Clinical_Pipeline" to analyze the text using ctakes clinical pipeline preprocessor."""
        messagebox.showinfo("Help", help_text)
    def process_general_NE_help(self):
        help_text = """Click to identify and categorizing general named entities."""
        messagebox.showinfo("Help", help_text)
    def chunk_parsing_help(self):
        help_text = """Click to represent how words in a sentence composed of words with specific grammar relationships."""
        messagebox.showinfo("Help", help_text)
    def constituency_parsing_help(self):
        help_text = """Click to represent how words combine to form phrases and how those phrases combine to create larger sentence structures."""
        messagebox.showinfo("Help", help_text)
    def dependency_parsing_help(self):
        help_text = """Click to represent relationships between words, where each word is connected to a governing word (head) through labeled directed edges, indicating the syntactic dependencies between them."""
        messagebox.showinfo("Help", help_text)
    def process_clinical_NE_help(self):
        help_text = """Click to identify and categorizing clinical named entities."""
        messagebox.showinfo("Help", help_text)
    def semantic_role_labeling_help(self):
        help_text = """Click to identify semantic roles, indicating how nouns and pronouns relate to the sentence's verb and contribute to its meaning."""
        messagebox.showinfo("Help", help_text)
    def negation_help(self):
        help_text = """Click to understand detected negated values."""
        messagebox.showinfo("Help", help_text)
    def polarity_help(self):
        help_text = """Click to understand polarity values detected by Ctakes."""
        messagebox.showinfo("Help", help_text)    
if __name__ == '__main__':
    # Create a new instance of the Tkinter root window
    root = tk.Tk()
    root.state('zoomed')
    # Create an instance of the TextViewer class, passing the root window as an argument
    text_viewer = TextViewer(root)
    # Start the Tkinter event loop to display the GUI and handle user interactions
    root.mainloop()