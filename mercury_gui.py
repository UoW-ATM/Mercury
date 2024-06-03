#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(1, '..')
sys.path.insert(1, 'libs/openap')

from dash import Dash, html, dcc
import dash
import logging
#logging.basicConfig(level=logging.DEBUG)


external_stylesheets = []
app = Dash(__name__,
           use_pages=True,
           assets_folder='dashboard/assets',
           pages_folder="dashboard/pages",
           external_stylesheets=external_stylesheets,
           suppress_callback_exceptions=True)

app.layout = html.Div([

	        html.Div(
            [

                html.Div(
                    [
                        html.H3(''),
                    ],
                    className="one-third column",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "Mercury Dashboard",
                                    style={"margin-bottom": "20px"},
                                ),

                            ]
                        )
                    ],
                    className="one-half column",
                    id="title",
                ),

                html.Div(
                    [

                        html.Img(src=dash.get_asset_url("uow.webp"),style={"height": "90px","width": "auto","margin-bottom": "5px",},)
                    ],
                    style={'text-align': 'right'},className="one-third column",
                ),
            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
        ),


    html.Div(['Pages:',dcc.Dropdown( options=[ { "label":dcc.Link(children=page['name'] ,href=page["relative_path"]), "value": page['name'] } for page in dash.page_registry.values()], value='Home', clearable=False)],style={"width": "10%"}),

	dash.page_container
])

if __name__ == '__main__':
	app.run_server(debug=True)
