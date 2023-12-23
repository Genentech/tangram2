
# This is necessary on certain machines to update the poetry config with the access tokens
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring

IMPORTS_USERNAME=$1
IMPORTS_TOKEN="glpat-qe1483sj-PxsjpmLoxN2"

poetry config repositories.gne_spaotsc https://code.roche.com/rb-aiml-cv-spatial/cci-explore/imported-methods/gne_spaotsc.git
poetry config http-basic.gne_spaotsc $IMPORTS_USERNAME $IMPORTS_TOKEN

poetry config repositories.gne_celery https://code.roche.com/rb-aiml-cv-spatial/cci-explore/imported-methods/gne-celery.git
poetry config http-basic.gne_celery $IMPORTS_USERNAME $IMPORTS_TOKEN

poetry config repositories.gne_tangram2 https://code.roche.com/tangramgroup/Tangram2.git
poetry config http-basic.gne_tangram2 $IMPORTS_USERNAME "glpat-q3r_6pd-zqTwiNHL9yXg"
