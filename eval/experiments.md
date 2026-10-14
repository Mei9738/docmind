| Change    | Why I tried it    | recall@1 | recall@3  | recall@5 | MRR       | Keep?    | Notes    |
| --------- | -------- | -------- | --------- | -------- | --------- | -------- | -------- |
| baseline	| — 	| 0.367	   | 0.633     | 0.800    | 0.511	  | —        |          |
| code-blockfix	| bug found while checking headings	| 0.367	   | 0.633     | 0.800    | 0.511	  | yes       |only affected debugging.md; no golden question uses that part; kept because headings are now correct          |
| chunk 500/100	| testing the chunk size knob	| 0.367	   | 0.667     | 0.733    | 0.500	  | no       |fixed q29 and q20, lost q19, q24, q28; smaller chunks fill the top 5 with pieces of the same section         |
| the heading in front of each chunk	| chunks in the middle of a section don’t know their topic (my Ticket #6 idea)	| 0.433	   | 0.767     | 0.833    | 0.578	  | -       |         |